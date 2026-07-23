import numpy as np

from tradingai.analysis.ml.baseline import BaselinePredictor
from tradingai.analysis.ml.sklearn_models import LogisticPredictor, RandomForestPredictor
from tradingai.analysis.ml.xgboost_model import XGBoostPredictor


def _synthetic_data(rows: int = 120):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(rows, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_baseline_predictor_fits_and_predicts():
    X, y = _synthetic_data()
    model = BaselinePredictor()
    model.fit(X, y)
    probabilities = model.predict_proba(X[:5])
    assert probabilities.shape == (5, 2)


def test_logistic_predictor_fits_and_predicts():
    X, y = _synthetic_data()
    model = LogisticPredictor()
    model.fit(X, y)
    probabilities = model.predict_proba(X[:5])
    assert probabilities.shape == (5, 2)


def test_random_forest_predictor_fits_and_predicts():
    X, y = _synthetic_data()
    model = RandomForestPredictor()
    model.fit(X, y)
    probabilities = model.predict_proba(X[:5])
    assert probabilities.shape == (5, 2)


def test_xgboost_predictor_fits_and_predicts():
    X, y = _synthetic_data()
    model = XGBoostPredictor()
    model.fit(X, y)
    probabilities = model.predict_proba(X[:5])
    assert probabilities.shape == (5, 2)
