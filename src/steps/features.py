"""Etapa 4 (features): verificação do Oldpeak negativo (R2) e etapa 5 (drop_cols).
Regras fixas, não aprendem nada.
drop_cols: cholesterol (tira o valor do colesterol), fastingbs (tira o FastingBS) ou
cholesterol_fastingbs (tira os dois), R2_atalho.

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


def drop_fastingbs():
    """Tira o FastingBS (glicemia de jejum > 120 mg/dL).

    Achado da análise de correlação: o FastingBS é a variável mais associada à ausência do
    colesterol (V de Cramér 0,36; 43% dos pacientes com FastingBS = 1 não têm colesterol
    registrado, contra 10% dos com FastingBS = 0). Parte da sua associação com o alvo pode vir do
    mesmo efeito de hospital de origem. R2_atalho__sem_fastingbs mede quanto ele contribui.
    """
    return DropCols(("FastingBS",))


def drop_cholesterol_fastingbs():
    """Tira o Cholesterol e o FastingBS juntos (R2_atalho, caminho C).

    No R2, sem_colesterol e sem_fastingbs empataram com o baseline cada um sozinho, mas a
    remoção dos dois juntos não tinha sido testada: o FastingBS pode compensar parte do que se
    perde com o colesterol (ou o contrário).
    """
    return DropCols(("Cholesterol", "FastingBS"))
