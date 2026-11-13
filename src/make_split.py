"""Prepara os dados: lê data/raw/heart.csv e gera os dois splits em data/split/.

  data/split/70_30/train.csv, test.csv   principal (R0-R10 e teste final)
  data/split/80_20/train.csv, test.csv   análise de sensibilidade (R11)

Preparação (feita ANTES de dividir):
  1. confere as colunas esperadas;
  2. remove linhas duplicadas (a mesma linha não pode cair no treino e no teste);
  3. divisão estratificada treino/teste, SEED fixa, uma por proporção.

Os zeros impossíveis (Cholesterol, RestingBP) NÃO são tratados aqui: o tratamento é uma etapa
do pipeline (zeros -> imputer), ajustada só no treino de cada fold. Aqui só são contados.
Os Oldpeak negativos ficam como estão (supradesnivelamento do ST: informação clínica real).

Os conjuntos de teste ficam isolados até o final_test.py.

Uso: python -m src.make_split [--force]
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FEATURES, RAW_CSV, SEED, SPLITS, TARGET, test_csv, train_csv


def load_raw():
    if not RAW_CSV.exists():
        raise SystemExit(
            f"{RAW_CSV} não existe. Baixe o heart.csv em "
            "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction "
            "e coloque em data/raw/.")
    df = pd.read_csv(RAW_CSV)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Colunas ausentes no heart.csv: {', '.join(missing)}")
    n0 = len(df)
    df = df[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
    print(f"Linhas no arquivo: {n0} | duplicatas removidas: {n0 - len(df)} | restantes: {len(df)}")
    return df


def report(df):
    """Contagem dos problemas conhecidos da base (para o artigo e para conferência)."""
    print("\nValores suspeitos (base inteira, antes de dividir):")
    for c in ["Cholesterol", "RestingBP"]:
        n = int((df[c] == 0).sum())
        print(f"  {c} = 0: {n} linhas ({n / len(df):.1%})  -> ausente codificado como 0")
    n = int((df["Oldpeak"] < 0).sum())
    print(f"  Oldpeak < 0: {n} linhas  -> mantidas (supradesnivelamento do ST)")
    prev = df[TARGET].mean()
    print(f"  Prevalência de {TARGET}=1: {prev:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regera os splits (invalida a comparação com execuções antigas)")
    args = ap.parse_args()

    if all(train_csv(s).exists() for s in SPLITS) and not args.force:
        print("Splits já existem em data/split/. Nada feito (use --force para regerar).")
        return

    df = load_raw()
    report(df)
    for split, test_size in SPLITS.items():
        train, test = train_test_split(df, test_size=test_size, random_state=SEED,
                                       stratify=df[TARGET])
        train_csv(split).parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_csv(split), index=False)
        test.to_csv(test_csv(split), index=False)
        print(f"\nsplit {split}: train {len(train)} linhas | test {len(test)} linhas | "
              f"positivos no treino {train[TARGET].mean():.1%}, no teste {test[TARGET].mean():.1%}")


if __name__ == "__main__":
    main()
