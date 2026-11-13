"""Mapa de calor C x gamma (R4): média da métrica na CV para cada par, num kernel.

Uso: python heatmap_c_gamma.py --rodada R4_rbf [--metric roc_auc] [--kernel rbf] [--incluir <pasta> ...]

Lê as execuções da rodada (nome começando com o prefixo), pega C e gamma da config (ausentes =
padrão do SVC: C=1.0, gamma="scale") e grava results/figuras/<rodada>_heatmap_<métrica>.png/.csv.
Leitura: canto de C pequeno = margem larga (mais erros tolerados, modelo mais simples); gamma
grande = cada ponto de treino influencia só a vizinhança (fronteira recortada, risco de
sobreajuste).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.config import FIGURAS_DIR, PRIMARY_METRIC, RUNS_DIR, SCORING


def gamma_key(g):
    """Ordena os gammas numéricos e põe 'scale'/'auto' no fim."""
    return (1, 0.0, str(g)) if isinstance(g, str) else (0, float(g), "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rodada", required=True, metavar="PREFIXO")
    ap.add_argument("--metric", default=PRIMARY_METRIC, choices=SCORING)
    ap.add_argument("--kernel", default="rbf")
    ap.add_argument("--incluir", nargs="+", default=[], metavar="PASTA",
                    help="execuções de fora da rodada que são um ponto da grade (ex.: o "
                         "baseline com C=1 e gamma=scale, pulado pelo hash)")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(RUNS_DIR.glob("*/")):
        meta_f = run_dir / "meta.json"
        if run_dir.name.startswith(".tmp_") or not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if not (meta["name"].startswith(args.rodada) or run_dir.name in args.incluir):
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        model = cfg["model"]
        if model["name"] != "svm" or model.get("kernel", "rbf") != args.kernel:
            continue
        folds = pd.read_csv(run_dir / "fold_scores.csv")
        rows.append({"C": float(model.get("C", 1.0)), "gamma": model.get("gamma", "scale"),
                     "media": folds[args.metric].mean(), "pct_sv": folds["pct_sv"].mean(),
                     "run": run_dir.name})
    if not rows:
        raise SystemExit(f"Nenhuma execução SVM {args.kernel} com prefixo '{args.rodada}'.")
    df = pd.DataFrame(rows).drop_duplicates(["C", "gamma"], keep="last")
    gammas = sorted(df["gamma"].unique(), key=gamma_key)
    grid = df.pivot(index="C", columns="gamma", values="media").reindex(columns=gammas)
    sv = df.pivot(index="C", columns="gamma", values="pct_sv").reindex(columns=gammas)

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.rodada.rstrip('_')}_heatmap_{args.metric}"
    df.to_csv(FIGURAS_DIR / f"{stem}.csv", index=False)

    vals = grid.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.3 * len(gammas) + 2.5, 0.8 * len(grid) + 2))
    im = ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(gammas)), [str(g) for g in gammas])
    ax.set_yticks(range(len(grid)), [f"{c:g}" for c in grid.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    best = np.nanmax(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isnan(vals[i, j]):
                continue
            txt = f"{vals[i, j]:.3f}\n{sv.iat[i, j]:.0f}% SV"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if vals[i, j] > np.nanmean(vals) else "white",
                    fontweight="bold" if vals[i, j] == best else "normal")
    fig.colorbar(im, ax=ax, label=f"{args.metric} (média da CV)")
    ax.set_title(f"SVM {args.kernel}: {args.metric} por C e gamma")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{stem}.png", dpi=130)
    plt.close(fig)

    with pd.option_context("display.width", 200):
        print(grid.to_string(float_format="%.3f"))
    if grid.isna().any().any():
        print("\nCélulas vazias: o ponto foi pulado pelo hash (tem a config de outra execução, "
              "em geral o baseline). Acrescente essa pasta com --incluir.")
    print(f"\n-> {FIGURAS_DIR / (stem + '.png')}")


if __name__ == "__main__":
    main()
