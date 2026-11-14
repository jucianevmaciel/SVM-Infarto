"""Figura didática: a fronteira do SVM em 2 dimensões (para a apresentação).

Uso:
  python fronteira_2d.py --run <pasta>                         # os hiperparâmetros da execução
  python fronteira_2d.py --run <pasta> --C 0.1 1 100           # um painel por C
  python fronteira_2d.py --run <pasta> --gamma 0.01 0.1 1      # um painel por gamma
  python fronteira_2d.py --run <pasta> --C 0.1 10 --gamma 0.01 1   # grade C x gamma

Como funciona: o pré-processamento da execução (zeros, imputação, one-hot, escalonamento...) é
ajustado no treino, os pacientes são projetados nas 2 primeiras componentes principais (PCA) e
um SVM com o mesmo kernel é treinado NESSE plano. Cada painel mostra:
  - fundo colorido: o lado da fronteira (azul = sem doença, vermelho = com doença);
  - linha cheia: a fronteira (decision_function = 0); tracejadas: as margens (= -1 e +1);
  - pontos com contorno preto: os vetores de suporte.

ATENÇÃO (dizer na apresentação): é uma ILUSTRAÇÃO. O modelo de verdade usa todas as colunas;
duas componentes guardam só parte da informação, então a fronteira e o número de vetores de
suporte aqui não são os do modelo avaliado. Só usa o treino (nunca o teste).

Grava em results/figuras/fronteira_2d_<rodada ou nome>.png.
"""
import argparse
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import src  # noqa: F401
from src.builder import build_pipeline
from src.config import FIGURAS_DIR, RUNS_DIR, SEED
from src.data import load_train
from src.registry import OPTIONAL_STEPS


def parse_gamma(v):
    try:
        return float(v)
    except ValueError:
        return v  # "scale" ou "auto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pasta em results/runs/ (de um SVM)")
    ap.add_argument("--C", nargs="+", type=float, help="valores de C, um painel por valor")
    ap.add_argument("--gamma", nargs="+", type=parse_gamma, help="valores de gamma")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run
    if not (run_dir / "config.yaml").exists():
        sys.exit(f"Execução '{args.run}' não encontrada em {RUNS_DIR}")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    name = cfg.pop("name", args.run)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    model_cfg = dict(cfg["model"])
    if model_cfg.pop("name") != "svm":
        sys.exit("A execução não é de um SVM.")

    # Pré-processamento da execução, sem calibração, limiar e modelo
    base = {k: v for k, v in cfg.items() if k not in OPTIONAL_STEPS}
    prep = Pipeline(build_pipeline(base).steps[:-1])
    X, y = load_train(meta.get("split", "70_30"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        Z = prep.fit_transform(X, y)
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    P = pca.transform(Z).to_numpy() if hasattr(Z, "columns") else pca.transform(Z)
    P = np.asarray(P)
    var = pca.explained_variance_ratio_

    Cs = args.C or [model_cfg.get("C", 1.0)]
    gammas = args.gamma or [model_cfg.get("gamma", "scale")]
    kernel = model_cfg.get("kernel", "rbf")
    if kernel == "linear" and args.gamma:
        print("Aviso: o kernel linear não usa gamma; os painéis de gamma serão iguais.")

    pad = 0.5
    xx, yy = np.meshgrid(np.linspace(P[:, 0].min() - pad, P[:, 0].max() + pad, 300),
                         np.linspace(P[:, 1].min() - pad, P[:, 1].max() + pad, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Só um dos dois variando: painéis lado a lado. Os dois: grade (linhas = gamma, colunas = C).
    if len(Cs) == 1 and len(gammas) > 1:
        layout = [[(0, j) for j in range(len(gammas))]]
        pairs = {(0, j): (Cs[0], gammas[j]) for j in range(len(gammas))}
    else:
        layout = [[(i, j) for j in range(len(Cs))] for i in range(len(gammas))]
        pairs = {(i, j): (Cs[j], gammas[i]) for i in range(len(gammas)) for j in range(len(Cs))}
    nrows, ncols = len(layout), len(layout[0])
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(max(4.6 * ncols, 9), 4.2 * nrows))
    colors = np.where(y.to_numpy() == 1, "#C44E52", "#4C72B0")
    for i in range(nrows):
        for j in range(ncols):
            c, g = pairs[(i, j)]
            params = {k: v for k, v in model_cfg.items() if k not in ("C", "gamma")}
            svm = SVC(C=c, gamma=g, random_state=SEED, **params).fit(P, y)
            D = svm.decision_function(grid).reshape(xx.shape)
            ax = axes[i][j]
            ax.contourf(xx, yy, D, levels=[-1e9, 0, 1e9], colors=["#4C72B0", "#C44E52"],
                        alpha=0.15)
            ax.contour(xx, yy, D, levels=[-1, 0, 1], colors="k",
                       linestyles=["--", "-", "--"], linewidths=[0.8, 1.6, 0.8])
            ax.scatter(P[:, 0], P[:, 1], c=colors, s=10, alpha=0.6, linewidths=0)
            sv = svm.support_
            ax.scatter(P[sv, 0], P[sv, 1], s=22, facecolors="none", edgecolors="k",
                       linewidths=0.6)
            acc = svm.score(P, y)
            g_txt = "" if kernel == "linear" else f", gamma={g}"
            ax.set_title(f"{kernel}: C={c:g}{g_txt}\n{len(sv)} vetores de suporte "
                         f"({100 * len(sv) / len(y):.0f}%), acerto no treino {acc:.0%}",
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"PC2 ({var[1]:.0%} da variância)")
            if i == nrows - 1:
                ax.set_xlabel(f"PC1 ({var[0]:.0%} da variância)")
    fig.suptitle("SVM projetado em 2 componentes principais (ilustração; só o treino)\n"
                 "azul = sem doença, vermelho = com doença; contorno preto = vetor de suporte",
                 fontsize=10)
    fig.tight_layout()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_C" if args.C else "") + ("_gamma" if args.gamma else "")
    out = FIGURAS_DIR / f"fronteira_2d_{name}{suffix}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"PCA: as 2 componentes guardam {var.sum():.0%} da variância das colunas tratadas.")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
