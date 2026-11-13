"""Etapa 8 (selection): seleção de atributos (R5). Atua nas colunas já codificadas e escalonadas.

  rfe_linear  eliminação recursiva com um SVM linear: tira, a cada passo, a coluna de menor
              |coeficiente|, até sobrarem n_features (Guyon et al., 2002, a proposta original
              do RFE é justamente com SVM)
  kbest       as k colunas de maior F (ANOVA) com o alvo; filtro simples, não usa o modelo
"""
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.svm import LinearSVC

from src.config import SEED


def rfe_linear(n_features=10, C=1.0):
    return RFE(LinearSVC(C=C, max_iter=20000, random_state=SEED),
               n_features_to_select=n_features)


def kbest(k=10):
    return SelectKBest(f_classif, k=k)
