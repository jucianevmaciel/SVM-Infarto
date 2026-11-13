import hashlib

import pandas as pd

from src.config import DEFAULT_SPLIT, SPLITS, TARGET, test_csv, train_csv


def _check_split(split):
    if split not in SPLITS:
        raise ValueError(f"split '{split}' não existe, opções: {', '.join(SPLITS)}")


def _load(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Coloque o heart.csv em data/raw/ e rode: python -m src.make_split")
    df = pd.read_csv(path)
    return df.drop(TARGET, axis=1), df[TARGET]


def load_train(split=DEFAULT_SPLIT):
    _check_split(split)
    return _load(train_csv(split))


def load_test(split=DEFAULT_SPLIT):  # SOMENTE final_test.py
    _check_split(split)
    return _load(test_csv(split))


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
