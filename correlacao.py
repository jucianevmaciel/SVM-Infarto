"""Análise exploratória: correlação entre as variáveis (rodar ANTES do R0).

Uso: python correlacao.py [--split 70_30] [--limiar 0.5]

Usa SÓ o treino do split (o teste continua isolado até o final_test.py). Nada aqui entra no
pipeline nem altera os dados: é uma leitura da base para orientar e justificar as rodadas.

Preparação (só para a análise, não grava nada nos CSVs):
  - Cholesterol = 0 e RestingBP = 0 viram ausentes (NaN), como no pipeline (zeros: to_nan);
  - Cholesterol_missing (0/1) é acrescentada, para medir o "atalho" do colesterol (R2_atalho).

Medidas (cada par recebe a medida adequada ao tipo das duas variáveis):
  contínua x contínua   Pearson (linear) e Spearman (monotônica, robusta a valores extremos)
  discreta x discreta   V de Cramér com correção de viés (Bergsma, 2013), de 0 a 1
  discreta x contínua   razão de correlação eta (quanto da variação da contínua é explicada
                        pelos grupos da discreta), de 0 a 1
  Contínuas: Age, RestingBP, Cholesterol, MaxHR, Oldpeak. Discretas: FastingBS (0/1), as 5
  categóricas, a flag do colesterol e o alvo HeartDisease.
  Na matriz mista, contínua x contínua entra como |Spearman| (todas as medidas de 0 a 1).
  A célula Cholesterol x Cholesterol_missing fica vazia: com a flag = 1, o colesterol é ausente.

Contra o alvo (HeartDisease), por variável:
  contínua   correlação ponto-bisserial (com sinal) e teste de Mann-Whitney (p-valor)
  discreta   V de Cramér e teste qui-quadrado (p-valor)
  todas      informação mútua (capta relações não lineares; em nats)

Leituras úteis para as rodadas:
  - Pares muito associados (>= --limiar) indicam redundância: interessam ao R4 (seleção) e à
    leitura das importâncias no R8 (duas colunas redundantes dividem a importância).
  - O ranking contra o alvo é o "esperado" do R8: ST_Slope, ChestPainType e Oldpeak no topo.
  - Cholesterol_missing x alvo e x Sex mostram o tamanho do atalho do colesterol (R2).
  - VIF das contínuas: multicolinearidade (VIF > 5 costuma ser considerado alto).

Grava em results/correlacao/<split>/:
  pearson.csv, spearman.csv           contínuas x contínuas (com sinal)
  associacao.csv                      matriz mista, todas as variáveis (0 a 1)
  alvo.csv                            cada variável contra o HeartDisease
  taxa_por_categoria.csv              % de doentes em cada categoria das discretas
  pares_fortes.csv                    pares com associação >= limiar
  vif.csv                             fator de inflação da variância das contínuas
  heatmap_numericas.png, heatmap_associacao.png, alvo.png
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")  # grava PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from src.config import (BINARY_NUM_COLS, CAT_COLS, CORRELACAO_DIR, DEFAULT_SPLIT, NUM_COLS,
                        SEED, SPLITS, TARGET, ZERO_COLS)
from src.data import load_train

FLAG = "Cholesterol_missing"
CONT_COLS = [c for c in NUM_COLS if c not in BINARY_NUM_COLS]  # 5 contínuas
DISC_COLS = BINARY_NUM_COLS + CAT_COLS + [FLAG]                # 7 discretas (sem o alvo)
FEATURES = CONT_COLS + DISC_COLS


# ------------------------------------------------------------------ preparação

def prepare(X, y):
    """Zeros impossíveis -> NaN e flag do colesterol, como no pipeline. Só para a análise."""
    df = X.copy()
    df[FLAG] = (df["Cholesterol"] == 0).astype(int)
    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
    df[TARGET] = y.to_numpy()
    return df


def is_cont(col):
    return col in CONT_COLS


# ------------------------------------------------------------------ medidas

def cramers_v(a, b):
    """V de Cramér com correção de viés (Bergsma, 2013). 0 = independentes, 1 = determinadas."""
    table = pd.crosstab(a, b)
    r, k = table.shape
    if r < 2 or k < 2:
        return np.nan
    n = table.to_numpy().sum()
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    den = min(k_c - 1, r_c - 1)
    return np.nan if den <= 0 else float(np.sqrt(phi2 / den))


def correlation_ratio(groups, values):
    """Razão de correlação eta: raiz da fração da variância de `values` explicada pelos grupos."""
    ok = values.notna() & groups.notna()
    g, v = groups[ok], values[ok].astype(float)
    if g.nunique() < 2:
        return np.nan  # ex.: Cholesterol x flag (com a flag = 1, o colesterol é sempre NaN)
    ss_total = ((v - v.mean()) ** 2).sum()
    ss_between = v.groupby(g).agg(lambda s: len(s) * (s.mean() - v.mean()) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan


def association(df, a, b):
    """Medida de 0 a 1 adequada ao tipo do par (usada na matriz mista)."""
    if is_cont(a) and is_cont(b):
        return abs(df[a].corr(df[b], method="spearman"))
    if not is_cont(a) and not is_cont(b):
        return cramers_v(df[a], df[b])
    cat, num = (a, b) if is_cont(b) else (b, a)
    return correlation_ratio(df[cat], df[num])


def mixed_matrix(df, cols):
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        m.loc[a, b] = m.loc[b, a] = association(df, a, b)
    return m


def vif(df):
    """VIF das contínuas: diagonal da inversa da matriz de correlação (casos completos)."""
    corr = df[CONT_COLS].dropna().corr().to_numpy()
    return pd.Series(np.diag(np.linalg.inv(corr)), index=CONT_COLS, name="vif")


def target_table(df):
    """Cada variável contra o alvo: medida, p-valor e informação mútua."""
    y = df[TARGET]
    rows = []
    for col in FEATURES:
        ok = df[col].notna()
        x, yy = df.loc[ok, col], y[ok]
        if is_cont(col):
            r, _ = stats.pointbiserialr(yy, x)
            p = stats.mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue
            mi = mutual_info_classif(x.to_frame(), yy, discrete_features=False,
                                     random_state=SEED)[0]
            rows.append({"variavel": col, "tipo": "contínua", "medida": "ponto-bisserial",
                         "valor": r, "forca": abs(r), "teste": "Mann-Whitney", "p_valor": p,
                         "info_mutua": mi, "n": int(ok.sum()),
                         "media_doentes": x[yy == 1].mean(),
                         "media_saudaveis": x[yy == 0].mean()})
        else:
            v = cramers_v(x, yy)
            p = stats.chi2_contingency(pd.crosstab(x, yy), correction=False)[1]
            codes = pd.factorize(x)[0].reshape(-1, 1)
            mi = mutual_info_classif(codes, yy, discrete_features=True, random_state=SEED)[0]
            rows.append({"variavel": col, "tipo": "discreta", "medida": "V de Cramér",
                         "valor": v, "forca": v, "teste": "qui-quadrado", "p_valor": p,
                         "info_mutua": mi, "n": int(ok.sum()),
                         "media_doentes": np.nan, "media_saudaveis": np.nan})
    return pd.DataFrame(rows).sort_values("forca", ascending=False).reset_index(drop=True)


def rate_table(df):
    """% de doentes e número de pacientes em cada categoria das variáveis discretas."""
    rows = []
    for col in DISC_COLS:
        g = df.groupby(col)[TARGET].agg(["size", "mean"])
        for cat, (n, rate) in g.iterrows():
            rows.append({"variavel": col, "categoria": cat, "n": int(n),
                         "pct_pacientes": 100 * n / len(df), "pct_doentes": 100 * rate})
    return pd.DataFrame(rows)


def strong_pairs(assoc, limiar):
    cols = [c for c in assoc.columns if c != TARGET]
    rows = [{"var_a": a, "var_b": b, "associacao": assoc.loc[a, b],
             "medida": ("|Spearman|" if is_cont(a) and is_cont(b) else
                        "V de Cramér" if not is_cont(a) and not is_cont(b) else "eta")}
            for a, b in itertools.combinations(cols, 2) if not np.isnan(assoc.loc[a, b])]
    pairs = pd.DataFrame(rows).sort_values("associacao", ascending=False).reset_index(drop=True)
    return pairs, pairs[pairs["associacao"] >= limiar]


# ------------------------------------------------------------------ figuras

def annotate(ax, m, fmt="{:.2f}", vmin=-1, vmax=1):
    vals = m.to_numpy(dtype=float)
    mid = (vmin + vmax) / 2
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isnan(vals[i, j]):
                continue
            strong = abs(vals[i, j] - mid) > 0.6 * (vmax - mid)
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=7.5,
                    color="white" if strong else "black")


def plot_numeric(pearson, spearman, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, m, title in [(axes[0], pearson, "Pearson (linear)"),
                         (axes[1], spearman, "Spearman (monotônica)")]:
        im = ax.imshow(m.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(m)), m.index)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        annotate(ax, m)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlação")
    fig.suptitle("Contínuas x contínuas (treino; zeros impossíveis como ausentes)")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_mixed(assoc, out):
    n = len(assoc)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.55 * n + 2))
    im = ax.imshow(assoc.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), assoc.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), assoc.index)
    annotate(ax, assoc, vmin=0, vmax=1)
    # Separa as contínuas das discretas e destaca o alvo
    k = len(CONT_COLS) - 0.5
    ax.axhline(k, color="white", lw=1.5)
    ax.axvline(k, color="white", lw=1.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="associação (0 a 1)")
    ax.set_title("Associação entre todas as variáveis (treino)\n"
                 "contínua x contínua: |Spearman|; discreta x discreta: V de Cramér; "
                 "mista: eta", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_target(tab, out):
    t = tab.iloc[::-1]
    colors = np.where(t["tipo"] == "contínua", "#4C72B0", "#DD8452")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(t) + 1.8), sharey=True)
    axes[0].barh(t["variavel"], t["forca"], color=colors)
    axes[0].set_xlabel("|ponto-bisserial| (contínuas) ou V de Cramér (discretas)")
    axes[0].set_title("Associação com HeartDisease")
    axes[1].barh(t["variavel"], t["info_mutua"], color=colors)
    axes[1].set_xlabel("informação mútua (nats)")
    axes[1].set_title("Informação mútua com HeartDisease")
    for ax in axes:
        ax.set_xlim(left=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#4C72B0", "#DD8452")]
    axes[1].legend(handles, ["contínua", "discreta"], loc="lower right", fontsize=8)
    fig.suptitle("Cada variável contra o alvo (treino); ordem: associação", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ principal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=DEFAULT_SPLIT, choices=list(SPLITS))
    ap.add_argument("--limiar", type=float, default=0.5,
                    help="associação a partir da qual um par é listado como forte (padrão 0,5)")
    args = ap.parse_args()

    X, y = load_train(args.split)  # SÓ o treino
    df = prepare(X, y)
    out = CORRELACAO_DIR / args.split
    out.mkdir(parents=True, exist_ok=True)

    pearson = df[CONT_COLS].corr(method="pearson")
    spearman = df[CONT_COLS].corr(method="spearman")
    assoc = mixed_matrix(df, FEATURES + [TARGET])
    tab = target_table(df)
    rates = rate_table(df)
    pairs, strong = strong_pairs(assoc, args.limiar)
    v = vif(df)

    pearson.to_csv(out / "pearson.csv", float_format="%.4f")
    spearman.to_csv(out / "spearman.csv", float_format="%.4f")
    assoc.to_csv(out / "associacao.csv", float_format="%.4f")
    tab.to_csv(out / "alvo.csv", index=False, float_format="%.4g")
    rates.to_csv(out / "taxa_por_categoria.csv", index=False, float_format="%.2f")
    strong.to_csv(out / "pares_fortes.csv", index=False, float_format="%.4f")
    v.to_csv(out / "vif.csv", float_format="%.3f")
    plot_numeric(pearson, spearman, out / "heatmap_numericas.png")
    plot_mixed(assoc, out / "heatmap_associacao.png")
    plot_target(tab, out / "alvo.png")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"Treino do split {args.split}: {len(df)} pacientes, "
              f"{df[TARGET].mean():.1%} com doença\n")
        print("Contra o alvo (ordenado pela força da associação):")
        print(tab[["variavel", "medida", "valor", "p_valor", "info_mutua", "n"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        print("\nPares mais associados entre si (sem o alvo):")
        print(pairs.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nPares com associação >= {args.limiar:g}: {len(strong)}")
        print("\nVIF das contínuas (> 5 costuma indicar multicolinearidade):")
        print(v.to_string(float_format=lambda x: f"{x:.2f}"))
        miss = rates[rates["variavel"] == FLAG].set_index("categoria")
        print(f"\nColesterol ausente: {miss.loc[1, 'n']} pacientes, "
              f"{miss.loc[1, 'pct_doentes']:.1f}% com doença "
              f"(registrado: {miss.loc[0, 'pct_doentes']:.1f}%)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
