"""config (dict) -> Pipeline do scikit-learn."""
import copy

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import DEFAULT_SPLIT, SPLITS
from src.registry import OPTIONAL_STEPS, REGISTRY, STEP_ORDER


class ConfigError(ValueError):
    pass


def normalize_step(value):
    """'standard' -> {'name': 'standard'};  {'name': 'rfe_linear', 'n_features': 8} fica igual."""
    if value is None:
        return {"name": "none"}
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, dict) and "name" in value:
        return dict(value)
    raise ConfigError(f"Formato inválido: {value!r}. Use 'nome' ou {{name: nome, ...}}")


def _same(a, b):
    """Bool não se mistura com número (True == 1 em Python)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def normalize_model(value):
    """O model é escrito como {name: svm, kernel: rbf, C: 10, gamma: 0.01, ...}.

    Sem `name`, vale svm. O nome sempre entra na config normalizada (e no hash).
    Parâmetros iguais ao padrão da fábrica (ex.: kernel: rbf, C: 1.0, gamma: scale) saem da
    config: o mesmo modelo tem sempre o mesmo hash, escrito com ou sem eles, e o runner não o
    executa duas vezes.
    """
    if value is None:
        value = {}
    if isinstance(value, str):
        value = {"name": value}
    if not isinstance(value, dict):
        raise ConfigError(f"model inválido: {value!r}. Use {{name: svm, ...}}")
    name = value.get("name", "svm")
    if name not in REGISTRY["model"]:
        raise ConfigError(f"model '{name}' não existe, opções: {', '.join(REGISTRY['model'])}")
    defaults = REGISTRY["model"][name]().get_params()
    params = {k: v for k, v in value.items()
              if k != "name" and not (k in defaults and _same(defaults[k], v))}
    return {"name": name, **params}


def resolve(baseline, override=None):
    """Junta baseline + override. O override SUBSTITUI a etapa inteira.

    `split` (70_30 ou 80_20) não é etapa: diz qual train.csv usar. O padrão (70_30) sai da
    config, como as etapas opcionais com none.
    """
    cfg = copy.deepcopy(baseline)
    for key, value in (override or {}).items():
        cfg[key] = copy.deepcopy(value)
    valid = STEP_ORDER + ["model"] + OPTIONAL_STEPS + ["split"]
    unknown = [k for k in cfg if k not in valid]
    if unknown:
        raise ConfigError(f"Etapa '{unknown[0]}' não existe. Etapas: {', '.join(valid)}")
    missing = [s for s in STEP_ORDER + ["model"] if s not in cfg]
    if missing:
        raise ConfigError(f"Config sem as etapas: {', '.join(missing)}")
    for step in STEP_ORDER + [s for s in OPTIONAL_STEPS if s in cfg]:
        cfg[step] = normalize_step(cfg[step])
    cfg["model"] = normalize_model(cfg["model"])
    # Opcional com none = não escrita (permite "tirar" o limiar ou a calibração num override)
    for step in OPTIONAL_STEPS:
        if step in cfg and cfg[step]["name"] == "none":
            if len(cfg[step]) > 1:
                raise ConfigError(f"{step} 'none' não aceita parâmetros: {cfg[step]}")
            del cfg[step]
    if "split" in cfg:
        # Sem aspas, o YAML lê 80_20 como o número 8020: aceita as duas formas
        cfg["split"] = {k.replace("_", ""): k for k in SPLITS}.get(str(cfg["split"]),
                                                                    str(cfg["split"]))
        if cfg["split"] not in SPLITS:
            raise ConfigError(f"split '{cfg['split']}' não existe, opções: {', '.join(SPLITS)}")
        if cfg["split"] == DEFAULT_SPLIT:
            del cfg["split"]
    # Ordem estável das chaves: o hash não depende da ordem do YAML
    order = STEP_ORDER + ["model"] + OPTIONAL_STEPS + ["split"]
    return {k: cfg[k] for k in order if k in cfg}


def split_of(cfg):
    return cfg.get("split", DEFAULT_SPLIT)


def _make(step, spec, *args):
    """Constrói o passo pelo REGISTRY; devolve None para 'none'."""
    params = dict(spec)
    name = params.pop("name")
    options = REGISTRY[step]
    if name not in options:
        raise ConfigError(f"{step} '{name}' não existe, opções: {', '.join(options)}")
    factory = options[name]
    if factory is None:
        if params:
            raise ConfigError(f"{step} 'none' não aceita parâmetros: {params}")
        return None
    try:
        return factory(*args, **params)
    except TypeError as e:
        raise ConfigError(f"{step} '{name}': parâmetros inválidos {params} ({e})") from e


def build_pipeline(cfg):
    if cfg["zeros"]["name"] != "none" and cfg["imputer"]["name"] == "none":
        raise ConfigError("zeros: to_nan exige um imputer (o SVM não aceita NaN)")
    if cfg["missing_flags"]["name"] != "none" and cfg["zeros"]["name"] == "none":
        raise ConfigError("missing_flags exige zeros: to_nan (sem NaN, a flag seria sempre 0)")
    steps = [(s, obj) for s in STEP_ORDER if (obj := _make(s, cfg[s])) is not None]
    model = _make("model", cfg["model"])
    # Hiperparâmetro com nome errado só apareceria no fit: confere já
    try:
        model.set_params(**{k: v for k, v in cfg["model"].items() if k != "name"})
        model._validate_params()
    except (ValueError, TypeError) as e:
        raise ConfigError(f"model: hiperparâmetro inválido ({e})") from e
    pipe = Pipeline(steps + [("model", model)])
    for step in OPTIONAL_STEPS:  # calibration, depois threshold
        if step in cfg:
            pipe = _make(step, cfg[step], pipe) or pipe
    return pipe


def unwrap(est):
    """Pipeline treinado, mesmo quando envolto pelo ajuste de limiar e/ou pela calibração."""
    while not isinstance(est, Pipeline):
        if hasattr(est, "calibrated_classifiers_"):     # CalibratedClassifierCV(ensemble=False)
            est = est.calibrated_classifiers_[0].estimator
        elif hasattr(est, "estimator_") and isinstance(est, BaseEstimator):  # TunedThreshold
            est = est.estimator_
        else:
            raise TypeError(f"Não sei desembrulhar {type(est).__name__}")
    return est


def model_of(fitted):
    """O modelo (último passo) de um pipeline treinado, embrulhado ou não."""
    return unwrap(fitted).named_steps["model"]


def svm_of(fitted):
    """O SVC de um pipeline treinado, ou None nos baselines."""
    m = model_of(fitted)
    return m if isinstance(m, SVC) else None
