"""Roda a validação cruzada de um experimento e grava numa pasta nova e imutável."""
import hashlib
import json
import platform
import shutil
import time
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")  # grava PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from joblib import parallel_config
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from sklearn.utils.parallel import Parallel, delayed  # repassa o set_config às threads

from src.builder import build_pipeline, model_of, split_of, svm_of
from src.config import (CV, N_REPEATS, N_SPLITS, PERM_REPEATS, PRIMARY_METRIC, RUNS_DIR,
                        SCORERS, SCORING, SEED, train_csv)
from src.data import file_sha256, load_train


def config_hash(cfg, train_sha):
    """Identidade do experimento: config completa + seed + CV + métricas + treino."""
    payload = {"config": cfg, "seed": SEED, "cv": [N_SPLITS, N_REPEATS],
               "scoring": SCORING, "train_sha256": train_sha}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def find_existing(full_hash):
    if not RUNS_DIR.exists():
        return None
    for meta in sorted(RUNS_DIR.glob("*/meta.json")):
        if json.loads(meta.read_text(encoding="utf-8")).get("hash") == full_hash:
            return meta.parent
    return None


# ------------------------------------------------------------------ por fold

def fold_importances(est, X_val, y_val, i):
    """Importância por permutação no fold de validação (pacientes que o modelo não viu).

    Embaralha cada coluna ORIGINAL (antes do one-hot: ChestPainType inteira, não cada dummy) e
    mede a queda do roc_auc. Passa pelo pipeline inteiro, então vale para qualquer kernel.
    """
    repeat, fold = divmod(i, N_SPLITS)
    perm = permutation_importance(est, X_val, y_val, scoring=SCORERS[PRIMARY_METRIC],
                                  n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
    return [{"repeat": repeat, "fold": fold, "feature": str(f), "importance": v}
            for f, v in zip(X_val.columns, perm.importances_mean)]


def fold_svm(est, n_train):
    """Vetores de suporte do fold: quantos e que fração do treino. NaN fora do SVM."""
    m = svm_of(est)
    if m is None:
        return {"n_sv": np.nan, "pct_sv": np.nan}
    n = int(m.n_support_.sum())
    return {"n_sv": n, "pct_sv": 100 * n / n_train}


# ------------------------------------------------------------------ artefatos do modelo final

def svm_stats(final_pipe, n_train):
    """Resumo do modelo final: kernel, hiperparâmetros, vetores de suporte, limiar."""
    model = model_of(final_pipe)
    stats = {"tipo": type(model).__name__}
    m = svm_of(final_pipe)
    if m is not None:
        stats.update({"kernel": m.kernel, "C": m.C, "gamma": m.gamma,
                      "gamma_efetivo": float(m._gamma) if m.kernel != "linear" else None,
                      "degree": m.degree if m.kernel == "poly" else None,
                      "n_features": int(m.n_features_in_),
                      "vetores_suporte": int(m.n_support_.sum()),
                      "vetores_suporte_por_classe": [int(v) for v in m.n_support_],
                      "pct_vetores_suporte": round(100 * m.n_support_.sum() / n_train, 1)})
    if hasattr(final_pipe, "best_threshold_"):
        stats["limiar"] = float(final_pipe.best_threshold_)
    return stats


def linear_coefs(final_pipe):
    """Coeficientes do SVM linear (ou da logreg), por coluna codificada. None nos outros."""
    model = model_of(final_pipe)
    if not hasattr(model, "coef_") or getattr(model, "kernel", "linear") != "linear":
        return None
    coef = np.ravel(model.coef_)
    df = pd.DataFrame({"coluna": model.feature_names_in_, "coeficiente": coef})
    return df.reindex(df["coeficiente"].abs().sort_values(ascending=False).index)


def save_model(final_pipe, out_dir, fold_imp, n_train):
    """Grava o pipeline treinado e o que dá para ler dele."""
    joblib.dump(final_pipe, out_dir / "pipeline.joblib", compress=3)
    (out_dir / "svm_stats.json").write_text(json.dumps(svm_stats(final_pipe, n_train), indent=2),
                                            encoding="utf-8")
    coefs = linear_coefs(final_pipe)
    if coefs is not None:
        coefs.to_csv(out_dir / "coeficientes_lineares.csv", index=False)

    perm = (fold_imp.groupby("feature")["importance"].agg(["mean", "std"])
            .sort_values("mean", ascending=False).reset_index())
    perm.to_csv(out_dir / "importances_permutation.csv", index=False)
    fold_imp.to_csv(out_dir / "fold_importances.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.45 * len(perm))))
    p = perm.iloc[::-1]
    ax.barh(p["feature"], p["mean"], xerr=p["std"], color="#4C72B0")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"queda do {PRIMARY_METRIC} ao embaralhar a coluna")
    ax.set_title(f"Importância por permutação, média dos {N_SPLITS * N_REPEATS} folds")
    fig.tight_layout()
    fig.savefig(out_dir / "importances.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ execução

def run(name, cfg, force=False, n_jobs=-1):
    split = split_of(cfg)
    tr_csv = train_csv(split)
    train_sha = file_sha256(tr_csv)
    full_hash = config_hash(cfg, train_sha)
    existing = find_existing(full_hash)
    if existing and not force:
        print(f"[{name}] já executado em {existing.name}, pulando (use --force para repetir)")
        return existing

    X, y = load_train(split)
    pipe = build_pipeline(cfg)
    t0 = time.time()
    # Threads em vez de processos: no Python 3.14 o joblib (loky) quebra quando o caminho do
    # projeto tem acento. Com ~640 pacientes, threads custam pouco.
    with warnings.catch_warnings(), parallel_config(backend="threading"):
        warnings.simplefilter("ignore", category=UserWarning)  # avisos repetidos por fold
        res = cross_validate(pipe, X, y, cv=CV, scoring=SCORERS, n_jobs=n_jobs,
                             return_estimator=True, return_indices=True)
        val_idx = res["indices"]["test"]
        tr_idx = res["indices"]["train"]
        imp_rows = Parallel(n_jobs=n_jobs)(
            delayed(fold_importances)(est, X.iloc[val], y.iloc[val], i)
            for i, (est, val) in enumerate(zip(res["estimator"], val_idx)))
    elapsed = time.time() - t0
    fold_imp = pd.DataFrame([r for rows in imp_rows for r in rows])
    sv = pd.DataFrame([fold_svm(est, len(tr)) for est, tr in zip(res["estimator"], tr_idx)])

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{full_hash[:6]}_{name}"
    final_dir = RUNS_DIR / run_id
    tmp_dir = RUNS_DIR / f".tmp_{run_id}"
    tmp_dir.mkdir(parents=True)
    try:
        (tmp_dir / "config.yaml").write_text(
            yaml.safe_dump({"name": name, **cfg}, sort_keys=False, allow_unicode=True),
            encoding="utf-8")

        # nanmean: brier e log_loss são NaN quando o modelo não dá probabilidade
        metrics = {}
        for m in SCORING:
            v = res[f"test_{m}"]
            metrics[m] = ({"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
                          if np.isfinite(v).any() else {"mean": None, "std": None})
        (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        n = len(res["fit_time"])
        folds = pd.DataFrame({"repeat": np.arange(n) // N_SPLITS, "fold": np.arange(n) % N_SPLITS,
                              **{m: res[f"test_{m}"] for m in SCORING},
                              **sv, "fit_time": res["fit_time"]})
        folds.to_csv(tmp_dir / "fold_scores.csv", index=False)

        # Modelo final: mesmo pipeline treinado no treino inteiro (os do CV são descartados)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            final_pipe = build_pipeline(cfg).fit(X, y)
        save_model(final_pipe, tmp_dir, fold_imp, len(X))

        meta = {"run_id": run_id, "name": name, "hash": full_hash,
                "date": datetime.now().isoformat(timespec="seconds"),
                "split": split, "train_sha256": train_sha, "n_train": int(len(X)),
                "seed": SEED, "cv": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})",
                "elapsed_s": round(elapsed, 2),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__}}
        if hasattr(final_pipe, "best_threshold_"):
            meta["threshold"] = float(final_pipe.best_threshold_)
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Só vira uma execução "válida" depois de tudo gravado
        for tentativa in range(3):
            try:
                tmp_dir.rename(final_dir)
                break
            except PermissionError:  # Windows: antivírus/indexador segurando a pasta
                if tentativa == 2:
                    raise
                time.sleep(1)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    show = ["roc_auc", "recall", "specificity", "accuracy"]
    summary = " | ".join(f"{m}={metrics[m]['mean']:.3f}±{metrics[m]['std']:.3f}" for m in show)
    tail = "" if sv["pct_sv"].isna().all() else f" | SV {sv['pct_sv'].mean():.0f}% do treino"
    print(f"[{name}] {summary}{tail} ({elapsed:.0f}s)  ->  {run_id}")
    return final_dir
