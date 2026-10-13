from datetime import datetime, timezone
import math
from unittest.mock import patch

from tradingai.predict.evaluation.metrics import compute_evaluation_metrics


def test_compute_evaluation_metrics_with_mocked_rows():
    rows = [
        {
            "probability_up": 0.7,
            "expected_return_pct": 1.0,
            "actual_return_pct": 2.5,
            "direction_correct": True,
            "combined_score": 0.8,
            "timestamp": datetime(2026, 1, 10, tzinfo=timezone.utc),
        }
    ]

    with (
        patch("tradingai.predict.evaluation.metrics.get_evaluated_rows", return_value=rows),
        patch("tradingai.predict.evaluation.metrics.get_segment_calibration", return_value=[]),
    ):
        metrics = compute_evaluation_metrics(5)

    horizon = metrics["horizons"]["5"]
    assert horizon["sample_size"] == 1
    assert horizon["direction_accuracy"] == 1.0
    assert horizon["precision"] == 1.0
    assert horizon["recall"] == 1.0
    assert horizon["brier_score"] is not None
    assert horizon["mae_expected_return_pct"] is not None


def test_compute_evaluation_metrics_ignores_nan_returns():
    rows = [
        {
            "probability_up": 0.7,
            "expected_return_pct": 1.0,
            "actual_return_pct": float("nan"),
            "direction_correct": False,
            "combined_score": 0.9,
            "timestamp": datetime(2026, 1, 10, tzinfo=timezone.utc),
        },
        {
            "probability_up": 0.7,
            "expected_return_pct": 1.0,
            "actual_return_pct": 2.0,
            "direction_correct": True,
            "combined_score": 0.8,
            "timestamp": datetime(2026, 1, 10, tzinfo=timezone.utc),
        },
    ]

    with (
        patch("tradingai.predict.evaluation.metrics.get_evaluated_rows", return_value=rows),
        patch("tradingai.predict.evaluation.metrics.get_segment_calibration", return_value=[]),
    ):
        metrics = compute_evaluation_metrics(3)

    horizon = metrics["horizons"]["3"]
    assert horizon["sample_size"] == 1
    assert horizon["mae_expected_return_pct"] == 1.0
    assert horizon["top_10_ranked_return_pct"] == 2.0
    assert math.isfinite(horizon["brier_score"])
