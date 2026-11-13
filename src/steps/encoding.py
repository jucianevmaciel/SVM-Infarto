"""Etapa 6 (encoding): as 5 colunas categóricas viram números. Obrigatória (o SVM só lê números).

  onehot   uma coluna 0/1 por categoria (ChestPainType_ASY, ST_Slope_Flat...). Categorias com
           dois valores (Sex, ExerciseAngina) viram uma única coluna (drop="if_binary").
           Categoria nova no transform vira tudo 0 (handle_unknown="ignore").
O encoder aprende as categorias no fit (treino do fold). As outras colunas passam intactas.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.config import CAT_COLS


def onehot():
    enc = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("cat", enc, CAT_COLS)], remainder="passthrough",
                             verbose_feature_names_out=False)
