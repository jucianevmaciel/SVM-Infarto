"""Etapas opcionais que envolvem o pipeline inteiro, nesta ordem:

1. calibration: transforma a saída do SVM em probabilidade (CalibratedClassifierCV) - R8;
2. threshold: escolhe o limiar de decisão (TunedThresholdClassifierCV) - R6.

As duas usam um CV interno feito só com o treino recebido (o fold de treino, na CV externa).
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV

from src.config import SEED

# Critérios aceitos para escolher o limiar. NÃO usar "recall": o limiar ótimo seria o mínimo
# (todo mundo positivo, recall = 1 e especificidade = 0).
#   f1                 equilíbrio entre precisão e recall
#   f2                 F-beta com beta = 2: o recall pesa 4x mais que a precisão
#   balanced_accuracy  média de sensibilidade e especificidade
THRESHOLD_SCORING = {"f1": "f1", "balanced_accuracy": "balanced_accuracy",
                     "f2": make_scorer(fbeta_score, beta=2)}


def _inner_cv(n_splits):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def calibrated(pipe, method, n_splits=5):
    """ensemble=False: o calibrador é ajustado em saídas fora da amostra (CV interno) e o SVM é
    treinado UMA vez no treino inteiro (o mesmo modelo da versão sem calibração)."""
    return CalibratedClassifierCV(pipe, method=method, cv=_inner_cv(n_splits), ensemble=False)


def sigmoid(pipe, n_splits=5):
    # Método de Platt (1999), proposto justamente para o SVM: uma curva logística sobre a margem.
    return calibrated(pipe, "sigmoid", n_splits)


def isotonic(pipe, n_splits=5):
    # Função em degraus, mais flexível; pode sobreajustar com poucos pacientes.
    return calibrated(pipe, "isotonic", n_splits)


def tuned(pipe, scoring="f2", n_splits=5):
    if scoring not in THRESHOLD_SCORING:
        raise TypeError(f"scoring '{scoring}' inválido, opções: {', '.join(THRESHOLD_SCORING)}")
    return TunedThresholdClassifierCV(pipe, scoring=THRESHOLD_SCORING[scoring],
                                      cv=_inner_cv(n_splits))
