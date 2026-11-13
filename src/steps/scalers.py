"""Etapa 7 (scaler): escalonamento das colunas numéricas contínuas.

O SVM mede distâncias (kernel RBF) e tamanhos de margem: sem escalonamento, Cholesterol
(~100 a 600) e MaxHR (~60 a 200) dominam Oldpeak (~-2 a 6). É a pergunta do R1.
Só as contínuas são escalonadas (Age, RestingBP, Cholesterol, MaxHR, Oldpeak). Dummies do
one-hot, flags e FastingBS já são 0/1 e ficam como estão.
  none      sem escalonamento (referência para medir o efeito)
  standard  média 0 e desvio 1
  minmax    reescala para [0, 1] (mesma faixa das colunas 0/1)
  robust    mediana e intervalo interquartil (menos sensível a valores extremos)
Todos aprendem os parâmetros no fit (só no treino do fold) e apenas aplicam no transform.
"""
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.config import BINARY_NUM_COLS, NUM_COLS

SCALE_COLS = [c for c in NUM_COLS if c not in BINARY_NUM_COLS]


class ColumnScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None):
        self.scaler = scaler

    def fit(self, X, y=None):
        self.cols_ = [c for c in SCALE_COLS if c in X.columns]
        self.scaler_ = clone(self.scaler).set_output(transform="default").fit(X[self.cols_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_] = self.scaler_.transform(X[self.cols_])
        return X


def standard():
    return ColumnScaler(StandardScaler())


def minmax():
    return ColumnScaler(MinMaxScaler())


def robust():
    return ColumnScaler(RobustScaler())
