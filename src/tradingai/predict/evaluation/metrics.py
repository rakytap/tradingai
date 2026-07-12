from __future__ import annotations

import statistics
from collections import defaultdict

from tradingai.predict.config import get_prediction_horizons
from tradingai.predict.storage.repository import get_evaluated_rows, get_segment_calibration


def _precision_recall(rows: list[dict]) -> dict[str, float | None]:
    tp = fp = fn = 0
    for row in rows:
        predicted_up = float(row["probability_up"]) >= 0.5
        actual_up = float(row["actual_return_pct"]) > 0
        if predicted_up and actual_up:
            tp += 1
        elif predicted_up and not actual_up:
            fp += 1
        elif not predicted_up and actual_up:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {"precision": precision, "recall": recall}


def _brier_score(rows: list[dict]) -> float | None:
    values: list[float] = []
    for row in rows:
        actual_up = 1.0 if float(row["actual_return_pct"]) > 0 else 0.0
        values.append((float(row["probability_up"]) - actual_up) ** 2)
    return statistics.mean(values) if values else None


def _mae_expected_return(rows: list[dict]) -> float | None:
    errors = [
        abs(float(row["expected_return_pct"]) - float(row["actual_return_pct"]))
        for row in rows
    ]
    return statistics.mean(errors) if errors else None


def _top_ranked_return(rows: list[dict], top_n: int = 10) -> float | None:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        ts = row["timestamp"]
        key = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
        by_date[key].append(row)

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
    summary: dict[str, dict] = {}

    for horizon in horizons:
        rows = get_evaluated_rows(horizon)
        if not rows:
            summary[str(horizon)] = {
                "sample_size": 0,
                "direction_accuracy": None,
                "precision": None,
                "recall": None,
                "mae_expected_return_pct": None,
                "brier_score": None,
                "top_10_ranked_return_pct": None,
            }
            continue

        direction_accuracy = statistics.mean(
            1.0 if row["direction_correct"] else 0.0 for row in rows
        )
        precision_recall = _precision_recall(rows)
        summary[str(horizon)] = {
            "sample_size": len(rows),
            "direction_accuracy": round(direction_accuracy, 4),
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
            "mae_expected_return_pct": (
                round(_mae_expected_return(rows), 4)
                if _mae_expected_return(rows) is not None
                else None
            ),
            "brier_score": (
                round(_brier_score(rows), 4) if _brier_score(rows) is not None else None
            ),
            "top_10_ranked_return_pct": (
                round(_top_ranked_return(rows), 4)
                if _top_ranked_return(rows) is not None
                else None
            ),
        }

    return {
        "horizons": summary,
        "segments": get_segment_calibration(horizon_days),
    }
