from __future__ import annotations

import math
import statistics
from collections import defaultdict

from tradingai.analysis.ml.config import get_direction_threshold
from tradingai.analysis.ml.direction import (
    actual_direction_up,
    direction_is_correct,
    predicted_direction_up,
)
from tradingai.predict.config import get_prediction_horizons
from tradingai.predict.storage.repository import get_evaluated_rows, get_segment_calibration


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rows_with_finite_returns(rows: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for row in rows:
        actual = _finite(row.get("actual_return_pct"))
        if actual is None:
            continue
        cleaned.append({**row, "actual_return_pct": actual})
    return cleaned


def _precision_recall(
    rows: list[dict],
    *,
    threshold: float | None = None,
) -> dict[str, float | None]:
    cutoff = get_direction_threshold() if threshold is None else threshold
    tp = fp = fn = 0
    for row in rows:
        predicted_up = predicted_direction_up(float(row["probability_up"]), threshold=cutoff)
        actual_up = actual_direction_up(float(row["actual_return_pct"]))
        if predicted_up and actual_up:
            tp += 1
        elif predicted_up and not actual_up:
            fp += 1
        elif not predicted_up and actual_up:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {"precision": precision, "recall": recall}


def _direction_accuracy(rows: list[dict], *, threshold: float | None = None) -> float | None:
    if not rows:
        return None
    cutoff = get_direction_threshold() if threshold is None else threshold
    correct = sum(
        1
        for row in rows
        if direction_is_correct(
            float(row["probability_up"]),
            float(row["actual_return_pct"]),
            threshold=cutoff,
        )
    )
    return correct / len(rows)


def _brier_score(rows: list[dict]) -> float | None:
    values: list[float] = []
    for row in rows:
        actual_up = 1.0 if actual_direction_up(float(row["actual_return_pct"])) else 0.0
        values.append((float(row["probability_up"]) - actual_up) ** 2)
    return statistics.mean(values) if values else None


def _mae_expected_return(rows: list[dict]) -> float | None:
    errors: list[float] = []
    for row in rows:
        expected = _finite(row.get("expected_return_pct"))
        actual = _finite(row.get("actual_return_pct"))
        if expected is None or actual is None:
            continue
        errors.append(abs(expected - actual))
    return statistics.mean(errors) if errors else None


def _top_ranked_return(rows: list[dict], top_n: int = 10) -> float | None:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        actual = _finite(row.get("actual_return_pct"))
        if actual is None:
            continue
        ts = row["timestamp"]
        key = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
        by_date[key].append({**row, "actual_return_pct": actual})

    daily_returns: list[float] = []
    for day_rows in by_date.values():
        ranked = sorted(day_rows, key=lambda item: float(item["combined_score"]), reverse=True)
        selected = ranked[:top_n]
        if not selected:
            continue
        daily_returns.append(
            statistics.mean(float(item["actual_return_pct"]) for item in selected)
        )
    return statistics.mean(daily_returns) if daily_returns else None


def compute_evaluation_metrics(horizon_days: int | None = None) -> dict:
    horizons = [horizon_days] if horizon_days is not None else list(get_prediction_horizons())
    threshold = get_direction_threshold()
    summary: dict[str, dict] = {}

    for horizon in horizons:
        rows = _rows_with_finite_returns(get_evaluated_rows(horizon))
        if not rows:
            summary[str(horizon)] = {
                "sample_size": 0,
                "direction_accuracy": None,
                "precision": None,
                "recall": None,
                "mae_expected_return_pct": None,
                "brier_score": None,
                "top_10_ranked_return_pct": None,
                "direction_threshold": round(threshold, 4),
            }
            continue

        direction_accuracy = _direction_accuracy(rows, threshold=threshold)
        precision_recall = _precision_recall(rows, threshold=threshold)
        mae = _mae_expected_return(rows)
        brier = _brier_score(rows)
        top10 = _top_ranked_return(rows)
        summary[str(horizon)] = {
            "sample_size": len(rows),
            "direction_accuracy": (
                round(direction_accuracy, 4) if direction_accuracy is not None else None
            ),
            "precision": (
                round(precision_recall["precision"], 4)
                if precision_recall["precision"] is not None
                else None
            ),
            "recall": (
                round(precision_recall["recall"], 4)
                if precision_recall["recall"] is not None
                else None
            ),
            "mae_expected_return_pct": round(mae, 4) if mae is not None else None,
            "brier_score": round(brier, 4) if brier is not None else None,
            "top_10_ranked_return_pct": round(top10, 4) if top10 is not None else None,
            "direction_threshold": round(threshold, 4),
        }

    return {
        "horizons": summary,
        "segments": get_segment_calibration(horizon_days),
        "direction_threshold": round(threshold, 4),
    }
