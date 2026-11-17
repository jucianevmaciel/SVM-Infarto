"""Catálogo: nome usado no YAML -> função que constrói o passo.

Método novo = função no arquivo da etapa + uma linha aqui.
"""
from src.steps import encoding, features, imputers, models, scalers, selection, wrappers, zeros

# Ordem fixa do pipeline. NÃO alterar por experimento.
# Tudo fica DENTRO do pipeline: cada etapa aprende só no treino de cada fold (anti-vazamento).
STEP_ORDER = ["zeros", "missing_flags", "imputer", "features", "drop_cols",
              "encoding", "scaler", "selection"]  # + "model" no fim
# Etapas que podem ser omitidas no YAML (equivalem a none). Envolvem o pipeline inteiro,
# nesta ordem: primeiro a calibração, depois o limiar (escolhido sobre a saída calibrada).
OPTIONAL_STEPS = ["calibration", "threshold"]

REGISTRY = {
    "zeros": {
        "none": None,
        "to_nan": zeros.to_nan,
    },
    "missing_flags": {
        "none": None,
        "flags": zeros.flags,
    },
    "imputer": {
        "none": None,
        "median": imputers.median,
        "median_by_group": imputers.median_by_group,
        "knn_scaled": imputers.knn_scaled,
        "iterative": imputers.iterative,
    },
    "features": {
        "none": None,
        "oldpeak_abs": features.oldpeak_abs,
        "st_elevation": features.st_elevation,
    },
    "drop_cols": {
        "none": None,
        "cholesterol": features.drop_cholesterol,
        "fastingbs": features.drop_fastingbs,
        "cholesterol_fastingbs": features.drop_cholesterol_fastingbs,
    },
    "encoding": {
        "onehot": encoding.onehot,
    },
    "scaler": {
        "none": None,
        "standard": scalers.standard,
        "minmax": scalers.minmax,
        "robust": scalers.robust,
    },
    "selection": {
        "none": None,
        "rfe_linear": selection.rfe_linear,
        "kbest": selection.kbest,
    },
    "model": {
        "svm": models.svm,
        "logreg": models.logreg,
        "gboost": models.gboost,
        "dummy": models.dummy,
    },
    "calibration": {
        "none": None,
        "sigmoid": wrappers.sigmoid,
        "isotonic": wrappers.isotonic,
    },
    "threshold": {
        "none": None,
        "tuned": wrappers.tuned,
    },
}
