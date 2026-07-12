from __future__ import annotations

import json
import statistics
from collections import defaultdict

from tradingai.predict.config import get_prediction_horizons
from tradingai.predict.storage.repository import get_evaluated_rows


def _precision_recall(rows: list[dict]) -> dict[str, float | None]:
    tp = fp = fn = 0
    for row in rows:
        horizons = json.loads(row["horizons_json"])
        horizon = next(
            (item for item in horizons if item["horizon_days"] == row["horizon_days"]),
            None,
        )
        if horizon is None:
            continue
        predicted_up = horizon["probability_up"] >= 0.5
        actual_up = row["actual_return_pct"] > 0
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
        horizons = json.loads(row["horizons_json"])
        horizon = next(
            (item for item in horizons if item["horizon_days"] == row["horizon_days"]),
            None,
        )
        if horizon is None:
            continue
        actual_up = 1.0 if row["actual_return_pct"] > 0 else 0.0
        values.append((horizon["probability_up"] - actual_up) ** 2)
    return statistics.mean(values) if values else None


def _mae_expected_return(rows: list[dict]) -> float | None:
    errors: list[float] = []
    for row in rows:
        horizons = json.loads(row["horizons_json"])
        horizon = next(
            (item for item in horizons if item["horizon_days"] == row["horizon_days"]),
            None,
        )
        if horizon is None:
            continue
        errors.append(abs(horizon["expected_return_pct"] - row["actual_return_pct"]))
    return statistics.mean(errors) if errors else None


def _top_ranked_return(rows: list[dict], top_n: int = 10) -> float | None:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_date[row["prediction_date"]].append(row)

    daily_returns: list[float] = []
    for day_rows in by_date.values():
        ranked = sorted(day_rows, key=lambda item: item["combined_score"], reverse=True)
        selected = ranked[:top_n]
        if not selected:
            continue
        daily_returns.append(
            statistics.mean(item["actual_return_pct"] for item in selected)
        )
    return statistics.mean(daily_returns) if daily_returns else None


def compute_evaluation_metrics(
    horizon_days: int | None = None,
    *,
    db_path: str | None = None,
) -> dict:
    horizons = [horizon_days] if horizon_days is not None else list(get_prediction_horizons())
    summary: dict[str, dict] = {}

    for horizon in horizons:
        rows = get_evaluated_rows(horizon, db_path=db_path)
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

        direction_accuracy = statistics.mean(row["direction_correct"] for row in rows)
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
                round(_brier_score(rows), 4)
                if _brier_score(rows) is not None
                else None
            ),
            "top_10_ranked_return_pct": (
                round(_top_ranked_return(rows), 4)
                if _top_ranked_return(rows) is not None
                else None
            ),
        }

    return {"horizons": summary}
