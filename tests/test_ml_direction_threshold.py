import numpy as np
import pytest

from tradingai.analysis.ml.config import get_direction_threshold
from tradingai.analysis.ml.direction import direction_is_correct, predicted_direction_up
from tradingai.analysis.ml.evaluation import evaluate_predictions


def test_get_direction_threshold_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRADINGAI_DIRECTION_THRESHOLD", raising=False)
    get_direction_threshold.cache_clear()
    assert get_direction_threshold() == 0.5


def test_predicted_direction_up_respects_threshold():
    assert predicted_direction_up(0.6, threshold=0.55) is True
    assert predicted_direction_up(0.6, threshold=0.65) is False


def test_direction_is_correct():
    assert direction_is_correct(0.7, 2.0, threshold=0.5) is True
    assert direction_is_correct(0.7, -1.0, threshold=0.5) is False
    assert direction_is_correct(0.4, -1.0, threshold=0.5) is True


def test_evaluate_predictions_uses_custom_threshold():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.6, 0.4, 0.55, 0.45])
    default_metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)
    strict_metrics = evaluate_predictions(y_true, y_prob, threshold=0.58)
    assert default_metrics["accuracy"] == 1.0
    assert strict_metrics["accuracy"] == 0.75
    assert strict_metrics["direction_threshold"] == 0.58
