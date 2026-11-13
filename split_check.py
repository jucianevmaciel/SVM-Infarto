"""R11 - O resultado final muda com o split? (70/30 x 80/20)

Uso: python split_check.py --run70 <pasta> --run80 <pasta>

As duas execuções devem ter a MESMA config (o r11_split_80_20.yaml repete a config final com
split: 80_20). Mostra lado a lado a CV de cada split e, se o final_test.py já foi rodado nas
duas, o teste. As notas NÃO são comparadas por teste t (treinos e folds diferentes): a leitura é
se a diferença entre os splits cabe na variação entre os folds (o desvio da CV).

Grava results/comparacoes/R11_splits.csv.
"""
import argparse
import json
import sys

import pandas as pd
import yaml

from src.config import COMPARACOES_DIR, FINAL_DIR, RUNS_DIR

METRICS = ["roc_auc", "recall", "specificity", "accuracy", "f1", "brier"]


def read(run):
    d = RUNS_DIR / run
    if not (d / "meta.json").exists():
        sys.exit(f"Execução '{run}' não encontrada em {RUNS_DIR}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    cfg.pop("name", None)
    cfg.pop("split", None)
    folds = pd.read_csv(d / "fold_scores.csv")
    final = FINAL_DIR / run / "metrics.json"
    test = json.loads(final.read_text(encoding="utf-8")) if final.exists() else None
    return meta, cfg, folds, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run70", required=True)
    ap.add_argument("--run80", required=True)
    args = ap.parse_args()

    m70, c70, f70, t70 = read(args.run70)
    m80, c80, f80, t80 = read(args.run80)
    if (m70.get("split"), m80.get("split")) != ("70_30", "80_20"):
        sys.exit(f"Splits inesperados: {m70.get('split')} e {m80.get('split')}")
    if c70 != c80:
        print("Atenção: as duas execuções NÃO têm a mesma config; a comparação perde o sentido.\n"
              f"  70/30: {c70}\n  80/20: {c80}\n")

    rows = []
    for m in METRICS:
        if f70[m].isna().all():
            continue
        r = {"métrica": m,
             "CV 70/30": f70[m].mean(), "desvio 70/30": f70[m].std(ddof=0),
             "CV 80/20": f80[m].mean(), "desvio 80/20": f80[m].std(ddof=0)}
        r["dif CV (80-70)"] = r["CV 80/20"] - r["CV 70/30"]
        r["dentro do desvio?"] = ("sim" if abs(r["dif CV (80-70)"]) <=
                                  max(r["desvio 70/30"], r["desvio 80/20"]) else "não")
        if t70 and t80:
            r["teste 70/30"], r["teste 80/20"] = t70.get(m), t80.get(m)
        rows.append(r)
    table = pd.DataFrame(rows)
    COMPARACOES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARACOES_DIR / "R11_splits.csv", index=False)
    with pd.option_context("display.width", 220):
        print(table.to_string(index=False, float_format="%.3f"))
    print(f"\nTreino: {m70['n_train']} (70/30) x {m80['n_train']} (80/20) pacientes")
    if not (t70 and t80):
        print("Teste: rode o final_test.py nas duas execuções para ver as colunas de teste.")
    print(f"-> {COMPARACOES_DIR / 'R11_splits.csv'}")


if __name__ == "__main__":
    main()
