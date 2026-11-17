from pathlib import Path

import numpy as np
from sklearn.metrics import (brier_score_loss, get_scorer, log_loss, make_scorer,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold

# Raiz do projeto calculada a partir deste arquivo: funciona local e no Colab
BASE = Path(__file__).resolve().parents[1]
RAW_CSV = BASE / "data/raw/heart.csv"
SPLIT_DIR = BASE / "data/split"
RUNS_DIR = BASE / "results/runs"
FINAL_DIR = BASE / "results/final"
SUMMARY_CSV = BASE / "results/summary.csv"
COMPARACOES_DIR = BASE / "results/comparacoes"
FIGURAS_DIR = BASE / "results/figuras"
ROBUSTEZ_DIR = BASE / "results/robustez"
CORRELACAO_DIR = BASE / "results/correlacao"

SEED = 42

# Dois splits treino/teste, gerados uma vez pelo make_split (mesma SEED, estratificados).
# As rodadas R1-R9 usam o principal (70/30, o mesmo dos trabalhos anteriores). O R10 refaz a
# configuração final no 80/20 (análise de sensibilidade: as conclusões se mantêm?).
SPLITS = {"70_30": 0.30, "80_20": 0.20}
DEFAULT_SPLIT = "70_30"


def train_csv(split=DEFAULT_SPLIT):
    return SPLIT_DIR / split / "train.csv"


def test_csv(split=DEFAULT_SPLIT):
    return SPLIT_DIR / split / "test.csv"


# heart.csv (fedesoriano, Kaggle): 918 pacientes, 11 atributos e o alvo HeartDisease (0/1)
TARGET = "HeartDisease"
NUM_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CAT_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = NUM_COLS + CAT_COLS
# Colunas em que 0 é fisiologicamente impossível => valor ausente codificado como 0
ZERO_COLS = ["Cholesterol", "RestingBP"]
# FastingBS já é 0/1 (glicemia de jejum > 120 mg/dL): não é padronizada, como as dummies
BINARY_NUM_COLS = ["FastingBS"]

# Validação cruzada do treino: estratificada, 5 folds x 5 repetições, sempre a mesma
# (mesmos folds em todas as execuções => as notas podem ser comparadas fold a fold).
N_SPLITS, N_REPEATS = 5, 5
CV = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)


def _proba1(est, X):
    """Probabilidade da classe 1, ou None quando o modelo não a fornece.

    O SVC só tem predict_proba com probability=True (ou dentro da calibração do R7). Sem
    probabilidade, brier e log_loss ficam NaN; o roc_auc usa o decision_function.
    """
    if not hasattr(est, "predict_proba"):
        return None
    return est.predict_proba(X)[:, 1]


def brier(est, X, y):
    """Brier score: erro quadrático médio da probabilidade prevista (menor é melhor)."""
    p = _proba1(est, X)
    return np.nan if p is None else brier_score_loss(y, p)


def logloss(est, X, y):
    """Log loss (entropia cruzada) da probabilidade prevista (menor é melhor)."""
    p = _proba1(est, X)
    return np.nan if p is None else log_loss(y, p, labels=[0, 1])


# recall = sensibilidade (doentes identificados); specificity = recall da classe 0 (saudáveis
# corretamente liberados). No contexto médico, o falso negativo é o erro mais caro.
SCORING = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy",
           "roc_auc", "brier", "log_loss"]
SCORERS = {m: get_scorer(m) for m in ["accuracy", "precision", "recall", "f1",
                                      "balanced_accuracy", "roc_auc"]}
SCORERS["specificity"] = make_scorer(recall_score, pos_label=0)
SCORERS.update({"brier": brier, "log_loss": logloss})
SCORERS = {m: SCORERS[m] for m in SCORING}
# Métricas em que menor é melhor (o compare.py e o summarize.py invertem a ordenação)
LOWER_IS_BETTER = {"brier", "log_loss"}
# Principal: roc_auc (como no DecisionTree e no RandomForest). O recall é a segunda métrica de
# decisão e aparece em todas as tabelas, junto com a matriz de confusão no teste final.
PRIMARY_METRIC = "roc_auc"
SECONDARY_METRIC = "recall"

# Importância por permutação: repetições por coluna, em cada fold de validação
PERM_REPEATS = 5

# Nível de significância do teste t corrigido (bilateral) usado pelo compare.py.
# Não entra no hash das execuções: mudar o ALPHA só muda a leitura das comparações.
ALPHA = 0.05
