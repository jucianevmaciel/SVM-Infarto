"""Avaliação de um pipeline treinado num conjunto de dados: métricas, matriz de confusão e curva
ROC. Usado pelo final_test.py."""
import matplotlib
matplotlib.use("Agg")  # grava PNG sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from src.config import SCORERS

LABELS = {0: "sem doença", 1: "com doença"}


def score_all(pipe, X, y):
    """Todas as métricas de config.SCORERS. Uma métrica que não pode ser calculada vira NaN."""
    out = {}
    for name, scorer in SCORERS.items():
        try:
            out[name] = float(scorer(pipe, X, y))
        except (ValueError, AttributeError) as e:
            out[name] = float("nan")
            print(f"Aviso: {name} não calculada ({e})")
    return out


def confusion(pipe, X, y):
    """Matriz de confusão (DataFrame real x predito), mais VP, FN, FP, VN nomeados."""
    cm = confusion_matrix(y, pipe.predict(X), labels=[0, 1])
    return pd.DataFrame(cm, index=[f"real_{LABELS[c]}" for c in (0, 1)],
                        columns=[f"pred_{LABELS[c]}" for c in (0, 1)])


def plot_confusion(cm, out, title):
    """Cor = proporção da classe real; número = contagem."""
    counts = cm.to_numpy()
    norm = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = [LABELS[0], LABELS[1]]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, label="proporção da classe real")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_roc(curves, out, title):
    """curves: lista de (rótulo, pipeline, X, y). Uma curva por modelo, mesmo gráfico."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, pipe, X, y in curves:
        RocCurveDisplay.from_estimator(pipe, X, y, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set_xlabel("1 - especificidade (falsos positivos)")
    ax.set_ylabel("sensibilidade / recall (verdadeiros positivos)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
