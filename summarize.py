"""Junta todas as execuções de results/runs/ numa tabela (sempre regenerada).

Uso: python summarize.py [--metric roc_auc] [--split 70_30]
"""
import argparse
import json

import pandas as pd
import yaml

from src.config import LOWER_IS_BETTER, PRIMARY_METRIC, RUNS_DIR, SCORING, SPLITS, SUMMARY_CSV
from src.registry import OPTIONAL_STEPS, STEP_ORDER


def describe(cfg):
    """Resumo curto: só as etapas diferentes de none.

    Ex.: 'to_nan > median > onehot > standard | svm(C=10,gamma=0.01) | threshold=tuned(scoring=f2)'.
    """
    def fmt(spec):
        params = dict(spec)
        name = params.pop("name")
        args = ",".join(f"{k}={v}" for k, v in params.items())
        return f"{name}({args})" if args else name

    parts = [fmt(cfg[s]) for s in STEP_ORDER if cfg[s]["name"] != "none"]
    extras = [f"{s}={fmt(cfg[s])}" for s in OPTIONAL_STEPS if s in cfg]
    return " > ".join(parts) + " | " + " | ".join([fmt(cfg["model"])] + extras)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default=PRIMARY_METRIC, choices=SCORING)
    ap.add_argument("--split", choices=list(SPLITS), help="só as execuções deste split")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(RUNS_DIR.glob("*/")):
        if run_dir.name.startswith(".tmp_") or not (run_dir / "meta.json").exists():
            continue
        meta = read_json(run_dir / "meta.json")
        metrics = read_json(run_dir / "metrics.json")
        stats = read_json(run_dir / "svm_stats.json")
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        row = {"run_id": meta["run_id"], "name": meta["name"], "date": meta["date"]}
        for m in SCORING:
            # None (modelo sem probabilidade) e métricas ausentes viram NaN
            for stat in ("mean", "std"):
                v = metrics.get(m, {}).get(stat)
                row[f"{m}_{stat}"] = float("nan") if v is None else v
        row["pct_sv"] = stats.get("pct_vetores_suporte", float("nan"))
        row["limiar"] = meta.get("threshold", float("nan"))
        row["split"] = meta.get("split", "?")
        row["segundos"] = meta.get("elapsed_s", float("nan"))
        row["pipeline"] = describe(cfg)
        rows.append(row)

    if not rows:
        print("Nenhuma execução em results/runs/.")
        return

    df = pd.DataFrame(rows).sort_values(f"{args.metric}_mean",
                                        ascending=args.metric in LOWER_IS_BETTER)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    if args.split:
        df = df[df["split"] == args.split]

    show = (["name", f"{args.metric}_mean", f"{args.metric}_std"]
            + [f"{m}_mean" for m in ["roc_auc", "recall", "specificity", "accuracy", "brier"]]
            + ["pct_sv", "split", "pipeline"])
    show = list(dict.fromkeys(show))
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        print(df[show].to_string(index=False, float_format="%.3f"))
    print(f"\n{len(df)} execuções mostradas -> {SUMMARY_CSV} (a tabela completa tem todas as "
          "métricas e todos os splits)")
    if df["split"].nunique() > 1:
        print("Atenção: execuções de splits diferentes usam treinos diferentes; compare as notas "
              "só dentro do mesmo split.")


if __name__ == "__main__":
    main()
