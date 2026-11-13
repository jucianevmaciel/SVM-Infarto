"""R10 - Robustez da configuração escolhida (só no treino, sem tocar no teste).

Uso: python robustez.py --run <pasta do vencedor> [--seeds 1 7 13 99 2024]

1. Outras sementes da CV: o SVM é determinístico, então a única fonte de acaso é a divisão
   dos folds. Refaz a CV 5x5 com outras sementes: a nota muda muito? (se a variação entre
   sementes for maior que as diferenças entre configs nas rodadas, as escolhas eram ruído)
2. Subgrupos: previsões fora da amostra da CV principal (cada paciente é previsto pelos modelos
   que não o viram) e métricas por subgrupo:
     Sex                M x F (a base tem ~79% de homens)
     colesterol         registrado x ausente (Cholesterol = 0 no arquivo, quase tudo da Suíça)
     idade              < 55 x >= 55
   Um modelo que acerta muito menos num subgrupo é um achado (e uma limitação) para o artigo.

Grava em results/robustez/<run>/: sementes.csv e subgrupos.csv.
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import parallel_config
from sklearn.base import clone
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.utils.parallel import Parallel, delayed

import src  # noqa: F401
from src.builder import build_pipeline
from src.config import (CV, N_REPEATS, N_SPLITS, ROBUSTEZ_DIR, RUNS_DIR, SCORERS)
from src.data import load_train


def scores_of(est, X):
    """Saída contínua para o roc_auc: probabilidade, se houver; senão, decision_function."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def oof_fold(pipe, X, y, tr, val, i):
    est = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
    return pd.DataFrame({"idx": val, "repeat": i // N_SPLITS,
                         "score": scores_of(est, X.iloc[val]),
                         "pred": est.predict(X.iloc[val])})


def subgroup_table(X, y, oof):
    groups = {
        "Sex": X["Sex"].map({"M": "M", "F": "F"}).fillna(X["Sex"].astype(str)),
        "colesterol": np.where(X["Cholesterol"] == 0, "ausente", "registrado"),
        "idade": np.where(X["Age"] >= 55, ">= 55", "< 55"),
    }
    rows = []
    for gname, g in groups.items():
        g = pd.Series(g, index=X.index)
        for level in sorted(g.unique()):
            accs = []
            for _, r in oof.groupby("repeat"):
                mask = g.iloc[r["idx"]].to_numpy() == level
                yt = y.iloc[r["idx"]].to_numpy()[mask]
                if len(np.unique(yt)) < 2:
                    continue
                accs.append({"roc_auc": roc_auc_score(yt, r["score"].to_numpy()[mask]),
                             "recall": recall_score(yt, r["pred"].to_numpy()[mask]),
                             "specificity": recall_score(yt, r["pred"].to_numpy()[mask],
                                                         pos_label=0)})
            a = pd.DataFrame(accs)
            n = int((g == level).sum())
            rows.append({"grupo": gname, "nivel": level, "pacientes": n,
                         "prevalencia": float(y[g == level].mean()),
                         **{f"{m}_media": a[m].mean() for m in a},
                         **{f"{m}_desvio": a[m].std(ddof=0) for m in a}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 13, 99, 2024])
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run
    if not (run_dir / "config.yaml").exists():
        sys.exit(f"Execução '{args.run}' não encontrada em {RUNS_DIR}")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg.pop("name", None)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    X, y = load_train(meta.get("split", "70_30"))
    pipe = build_pipeline(cfg)
    out_dir = ROBUSTEZ_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    scoring = {m: SCORERS[m] for m in ["roc_auc", "recall", "specificity", "accuracy"]}

    with warnings.catch_warnings(), parallel_config(backend="threading"):
        warnings.simplefilter("ignore", category=UserWarning)
        # 1. sementes
        seed_rows = [{"semente": 42, "origem": "CV principal",
                      **{m: pd.read_csv(run_dir / "fold_scores.csv")[m].mean() for m in scoring}}]
        for s in args.seeds:
            cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=s)
            res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=args.n_jobs)
            seed_rows.append({"semente": s, "origem": "outra semente",
                              **{m: res[f"test_{m}"].mean() for m in scoring}})
            print(f"semente {s}: roc_auc {res['test_roc_auc'].mean():.3f} | "
                  f"recall {res['test_recall'].mean():.3f}")
        # 2. subgrupos (CV principal)
        parts = Parallel(n_jobs=args.n_jobs)(
            delayed(oof_fold)(pipe, X, y, tr, val, i)
            for i, (tr, val) in enumerate(CV.split(X, y)))
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(out_dir / "sementes.csv", index=False)
    sub = subgroup_table(X, y, pd.concat(parts, ignore_index=True))
    sub.to_csv(out_dir / "subgrupos.csv", index=False)

    with pd.option_context("display.width", 200):
        print("\nSementes da CV:\n" + seeds.to_string(index=False, float_format="%.3f"))
        print(f"amplitude do roc_auc entre sementes: {seeds['roc_auc'].max() - seeds['roc_auc'].min():.3f}"
              f" | do recall: {seeds['recall'].max() - seeds['recall'].min():.3f}")
        print("\nSubgrupos (previsões fora da amostra, média das repetições):\n"
              + sub.to_string(index=False, float_format="%.3f"))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
