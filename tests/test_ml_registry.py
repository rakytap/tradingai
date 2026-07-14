import pytest

from tradingai.analysis.ml.registry import get_predictor, list_predictors


def test_list_predictors_includes_all_models():
    models = list_predictors()
    assert models == ["baseline", "logistic", "random_forest", "xgboost"]


def test_get_predictor_returns_named_instance():
    model = get_predictor("logistic")
    assert model.name == "logistic"


def test_get_predictor_unknown_raises():
    with pytest.raises(ValueError, match="Unknown ML model"):
        get_predictor("unknown_model")
