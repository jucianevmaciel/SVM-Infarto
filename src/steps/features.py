"""Etapa 4 (features): verificação do Oldpeak negativo (R2) e etapa 5 (drop_cols).
Regras fixas, não aprendem nada.

Oldpeak é o desnível do segmento ST no esforço. Positivo = infradesnivelamento (o comum);
negativo = SUPRAdesnivelamento, sinal clássico de infarto. Por isso as linhas NÃO são removidas.
  none          mantém o valor como está (hipótese do grupo: é o melhor ou empata)
  oldpeak_abs   usa |Oldpeak| (trata supra e infra como "desvio", perde o sentido)
  st_elevation  mantém o Oldpeak e acrescenta ST_elevation = 1 quando Oldpeak < 0
"""
from sklearn.base import BaseEstimator, TransformerMixin


class OldpeakAbs(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Oldpeak"] = X["Oldpeak"].abs()
        return X


class StElevation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["ST_elevation"] = (X["Oldpeak"] < 0).astype(int)
        return X


def oldpeak_abs():
    return OldpeakAbs()


def st_elevation():
    return StElevation()


# ------------------------------------------------------------------ etapa drop_cols (R2)

class DropCols(BaseEstimator, TransformerMixin):
    """Remove colunas (as que existirem). Não aprende nada dos dados."""

    def __init__(self, cols=("Cholesterol",)):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


def drop_cholesterol():
    """Tira o valor do Cholesterol. A flag Cholesterol_missing, se houver, continua.

    Serve para separar dois efeitos (R2_atalho): o colesterol como medida clínica e o fato de
    ele estar ausente. Na base, Cholesterol = 0 tem 88% de doentes contra 48% nos registrados,
    porque os zeros vêm quase todos de um único hospital (Suíça). A flag pode subir o roc_auc
    por um atalho ("veio daquele hospital"), não por fisiologia.
    """
    return DropCols(("Cholesterol",))
