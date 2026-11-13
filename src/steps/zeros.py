"""Etapa 1 (zeros) e etapa 2 (missing_flags).

Cholesterol = 0 (~170 pacientes, quase todos da base suíça) e RestingBP = 0 (1 paciente) são
biologicamente impossíveis: são exames não registrados, codificados como 0. Deixar o 0 cru faz
o modelo aprender um padrão falso ("colesterol baixo" = mais doença).

  zeros: to_nan          troca o 0 por NaN em ZERO_COLS (o imputer preenche depois)
  missing_flags: flags   acrescenta <col>_missing (0/1) ANTES da imputação. Por padrão só para
                         o Cholesterol: o RestingBP tem 1 caso, a coluna seria quase constante.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import ZERO_COLS


class ZeroToNan(BaseEstimator, TransformerMixin):
    def __init__(self, cols=tuple(ZERO_COLS)):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.cols)
        X[cols] = X[cols].replace(0, np.nan)
        return X


class AddMissingFlags(BaseEstimator, TransformerMixin):
    """Acrescenta <col>_missing (0/1). Precisa vir ANTES da imputação."""

    def __init__(self, cols=("Cholesterol",)):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[f"{c}_missing"] = X[c].isna().astype(int)
        return X


def to_nan():
    return ZeroToNan()


def flags(cols=("Cholesterol",)):
    return AddMissingFlags(tuple(cols))
