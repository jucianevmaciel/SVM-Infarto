"""Etapa final (model): o SVM e os modelos de referência do R0.

Os parâmetros escritos no YAML vão direto para o construtor. Parâmetros iguais ao padrão da
fábrica saem da config (mesmo modelo = mesmo hash; ver builder.normalize_model).

  svm      SVC (kernel rbf, C=1, gamma="scale" no padrão do scikit-learn). Sem predict_proba:
           o roc_auc usa o decision_function; probabilidades só com a calibração (R7).
  logreg   regressão logística: o "modelo simples" (fronteira linear)
  gboost   gradient boosting: a referência forte sugerida pelo grupo
  dummy    sempre a classe mais frequente do treino (piso)
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.config import SEED


def svm(**params):
    params.setdefault("random_state", SEED)
    return SVC(**params)


def logreg(**params):
    params.setdefault("random_state", SEED)
    params.setdefault("max_iter", 2000)
    return LogisticRegression(**params)


def gboost(**params):
    params.setdefault("random_state", SEED)
    return GradientBoostingClassifier(**params)


def dummy(**params):
    params.setdefault("strategy", "most_frequent")
    params.setdefault("random_state", SEED)
    return DummyClassifier(**params)
