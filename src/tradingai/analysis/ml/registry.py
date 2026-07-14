from __future__ import annotations

from tradingai.analysis.ml.base import DirectionPredictor
from tradingai.analysis.ml.baseline import BaselinePredictor
from tradingai.analysis.ml.sklearn_models import LogisticPredictor, RandomForestPredictor
from tradingai.analysis.ml.xgboost_model import XGBoostPredictor

_MODELS: dict[str, type] = {
    "baseline": BaselinePredictor,
    "logistic": LogisticPredictor,
    "random_forest": RandomForestPredictor,
    "xgboost": XGBoostPredictor,
}


def list_predictors() -> list[str]:
    return sorted(_MODELS)


def get_predictor(name: str) -> DirectionPredictor:
    normalized = name.lower().strip()
    predictor_cls = _MODELS.get(normalized)
    if predictor_cls is None:
        supported = ", ".join(sorted(_MODELS))
        raise ValueError(
            f"Unknown ML model '{name}'. Supported models: {supported}."
        )
    return predictor_cls()
