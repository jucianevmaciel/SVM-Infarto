"""Avaliação final no conjunto de teste (rodar só no fim, uma vez, na execução escolhida).

Uso:
  python final_test.py --run <pasta do SVM escolhido> [--baselines <pasta R0> <pasta R0> ...]

O split (70_30 ou 80_20) vem do meta.json da execução: o R11 usa este mesmo script na execução
feita no 80/20. Os baselines precisam ser do MESMO split (o script confere).

Grava em results/final/<run>/: metrics.json, confusion_matrix.csv e .png, roc_curve.png
(SVM e baselines na mesma figura), baselines.csv e meta.json.
"""
import argparse
import json
import sys
import warnings
from datetime import datetime

import joblib
import pandas as pd
import yaml

import src  # noqa: F401  aplica set_config(transform_output="pandas") antes de carregar o joblib
from src.builder import build_pipeline
from src.config import FINAL_DIR, RUNS_DIR
from src.data import load_test, load_train
from src.evaluate import confusion, plot_confusion, plot_roc, score_all


def load_run(name):
    run_dir = RUNS_DIR / name
    if not (run_dir / "config.yaml").exists():
        sys.exit(f"Execução '{name}' não encontrada em {RUNS_DIR}")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    label = cfg.pop("name", name)
    if (run_dir / "pipeline.joblib").exists():
        pipe = joblib.load(run_dir / "pipeline.joblib")   # exatamente o modelo da execução
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pipe = build_pipeline(cfg).fit(*load_train(meta.get("split", "70_30")))
    return label, meta, pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="nome da pasta em results/runs/")
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="pastas do R0 avaliadas no mesmo teste, para comparação")
    args = ap.parse_args()

    out_dir = FINAL_DIR / args.run
    if out_dir.exists():
        print(f"Já avaliado no teste. Resultado guardado em {out_dir}:")
        print((out_dir / "metrics.json").read_text(encoding="utf-8"))
        return

    label, meta, pipe = load_run(args.run)
    split = meta.get("split", "70_30")
    X_test, y_test = load_test(split)

    metrics = score_all(pipe, X_test, y_test)
    cm = confusion(pipe, X_test, y_test)
    curves = [(f"SVM ({label})", pipe, X_test, y_test)]
    base_rows = [{"modelo": label, **metrics}]
    for b in args.baselines:
        b_label, b_meta, b_pipe = load_run(b)
        if b_meta.get("split", "70_30") != split:
            sys.exit(f"O baseline '{b}' é do split {b_meta.get('split')}, o SVM é do {split}.")
        base_rows.append({"modelo": b_label, **score_all(b_pipe, X_test, y_test)})
        if b_label.startswith("R0_dummy"):
            continue  # a curva do dummy é a diagonal
        curves.append((b_label, b_pipe, X_test, y_test))
    table = pd.DataFrame(base_rows).set_index("modelo")

    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    table.to_csv(out_dir / "baselines.csv")
    cm.to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion(cm, out_dir / "confusion_matrix.png",
                   f"Teste {split.replace('_', '/')} ({len(y_test)} pacientes)")
    plot_roc(curves, out_dir / "roc_curve.png", f"Curva ROC no teste ({split.replace('_', '/')})")
    fn, tp = int(cm.iloc[1, 0]), int(cm.iloc[1, 1])
    meta_out = {"run": args.run, "date": datetime.now().isoformat(timespec="seconds"),
                "split": split, "n_test": len(X_test), "baselines": args.baselines,
                "doentes_no_teste": int(y_test.sum()), "doentes_identificados": tp,
                "doentes_perdidos (falsos negativos)": fn}
    if "threshold" in meta:
        meta_out["limiar"] = meta["threshold"]
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    with pd.option_context("display.width", 200):
        print(table.T.to_string(float_format="%.3f"))
    print(f"\nMatriz de confusão:\n{cm.to_string()}")
    print(f"\nDoentes no teste: {int(y_test.sum())} | identificados: {tp} | perdidos: {fn}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
