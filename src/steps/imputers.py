"""Etapa 3 (imputer). Só atua nas colunas numéricas que têm NaN (as categóricas não têm).

Todos aprendem os valores no fit (treino do fold) e só aplicam no transform (anti-vazamento).
  median          mediana da coluna
  median_by_group mediana dentro de um grupo de ATRIBUTO (Sex, por padrão; ou faixa de idade).
                  NUNCA agrupar pelo alvo HeartDisease: o rótulo vazaria para dentro do atributo.
  knn_scaled      k vizinhos mais próximos, com as colunas numéricas padronizadas para medir a
                  distância (senão Cholesterol e MaxHR dominam)
  iterative       imputação iterativa (MICE): uma regressão prevê cada coluna com ausentes a
                  partir das outras numéricas, repetindo até estabilizar
Ordem de simplicidade (desempate): median -> median_by_group -> knn_scaled -> iterative.
O SVM não aceita NaN: com zeros: to_nan, algum imputer é obrigatório (o builder confere).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (habilita o IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

from src.config import NUM_COLS, SEED

AGE_BINS = [-np.inf, 45, 55, 65, np.inf]


def _num_cols(X):
    return [c for c in NUM_COLS if c in X.columns]


def _groups(X, group):
    if group == "age_band":
        return pd.cut(X["Age"], AGE_BINS, labels=False).astype(str)
    if group not in X.columns:
        raise ValueError(f"grupo '{group}' não é uma coluna da base")
    return X[group].astype(str)


class MedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X[_num_cols(X)].median()
        return self

    def transform(self, X):
        X = X.copy()
        cols = _num_cols(X)
        X[cols] = X[cols].fillna(self.medians_)
        return X


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Mediana por grupo; grupo sem valor observado (ou novo no transform) usa a mediana geral."""

    def __init__(self, group="Sex"):
        self.group = group

    def fit(self, X, y=None):
        cols = _num_cols(X)
        self.global_ = X[cols].median()
        self.by_group_ = X[cols].groupby(_groups(X, self.group)).median()
        return self

    def transform(self, X):
        X = X.copy()
        cols = _num_cols(X)
        g = _groups(X, self.group)
        fill = self.by_group_.reindex(g.to_numpy()).set_axis(X.index)
        fill = fill.fillna(self.global_)
        X[cols] = X[cols].fillna(fill[cols])
        return X


class KNNScaledImputer(BaseEstimator, TransformerMixin):
    """KNN nas colunas numéricas padronizadas; só as células ausentes são trocadas."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        num = X[_num_cols(X)]
        self.scaler_ = StandardScaler().set_output(transform="default").fit(num)
        self.knn_ = (KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True)
                     .set_output(transform="default").fit(self.scaler_.transform(num)))
        return self

    def transform(self, X):
        X = X.copy()
        cols = _num_cols(X)
        num = X[cols]
        back = self.scaler_.inverse_transform(self.knn_.transform(self.scaler_.transform(num)))
        out = num.to_numpy(dtype=float, copy=True)
        mask = num.isna().to_numpy()
        out[mask] = back[mask]
        X[cols] = out
        return X


class IterativeNumImputer(BaseEstimator, TransformerMixin):
    """MICE nas colunas numéricas: cada coluna com ausentes é prevista por uma regressão linear
    bayesiana (BayesianRidge) a partir das outras, em rodadas, até estabilizar. Só as células
    ausentes são trocadas. Usa só as numéricas (as categóricas ainda são texto nesta etapa)."""

    def __init__(self, max_iter=10):
        self.max_iter = max_iter

    def fit(self, X, y=None):
        self.imp_ = (IterativeImputer(max_iter=self.max_iter, random_state=SEED,
                                      keep_empty_features=True)
                     .set_output(transform="default").fit(X[_num_cols(X)]))
        return self

    def transform(self, X):
        X = X.copy()
        cols = _num_cols(X)
        num = X[cols]
        out = num.to_numpy(dtype=float, copy=True)
        mask = num.isna().to_numpy()
        out[mask] = self.imp_.transform(num)[mask]
        X[cols] = out
        return X


def median():
    return MedianImputer()


def median_by_group(group="Sex"):
    return GroupMedianImputer(group)


def knn_scaled(n_neighbors=5):
    return KNNScaledImputer(n_neighbors)


def iterative(max_iter=10):
    return IterativeNumImputer(max_iter)
