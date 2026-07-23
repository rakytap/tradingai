from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any

from tradingai.analysis.ml.config import get_direction_threshold
from tradingai.analysis.ml.direction import direction_is_correct
from tradingai.predict.config import DEFAULT_WEIGHTS, HORIZON_SCALES
from tradingai.predict.fusion.scorer import _sigmoid

FusionWeights = tuple[float, float, float]
MIN_AGENT_WEIGHT = 0.10
DEFAULT_SCALE_CANDIDATES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)


class FusionOptimizationError(ValueError):
    """Raised when fusion tuning cannot run (e.g. insufficient evaluated rows)."""


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def grid_weight_candidates(
    *,
    step: float = 0.05,
    min_weight: float = MIN_AGENT_WEIGHT,
) -> list[FusionWeights]:
    candidates: list[FusionWeights] = []
    for w_technical in _frange(min_weight, 1.0 - 2 * min_weight, step):
        for w_news in _frange(min_weight, 1.0 - w_technical - min_weight, step):
            w_regime = round(1.0 - w_technical - w_news, 4)
            if w_regime + 1e-9 >= min_weight:
                candidates.append((w_technical, w_news, w_regime))
    return candidates


def grid_scale_candidates() -> list[float]:
    return list(DEFAULT_SCALE_CANDIDATES)


def combined_score(row: dict[str, Any], weights: FusionWeights) -> float:
    w_technical, w_news, w_regime = weights
    return (
        w_technical * float(row["technical_score"])
        + w_news * float(row["news_score"])
        + w_regime * float(row["regime_score"])
    )


def probabilities_for_rows(
    rows: list[dict[str, Any]],
    weights: FusionWeights,
    scale: float,
) -> list[float]:
    return [_sigmoid(combined_score(row, weights) * scale) for row in rows]


def brier_score(probabilities: list[float], actual_returns_pct: list[float]) -> float:
    if not probabilities:
        raise ValueError("Cannot compute Brier score on empty data")
    values = [
        (prob - (1.0 if float(actual) > 0 else 0.0)) ** 2
        for prob, actual in zip(probabilities, actual_returns_pct, strict=True)
    ]
    return statistics.mean(values)


def direction_accuracy(
    probabilities: list[float],
    actual_returns_pct: list[float],
    *,
    threshold: float | None = None,
) -> float:
    if not probabilities:
        raise ValueError("Cannot compute direction accuracy on empty data")
    cutoff = get_direction_threshold() if threshold is None else threshold
    correct = sum(
        1
        for prob, actual in zip(probabilities, actual_returns_pct, strict=True)
        if direction_is_correct(prob, actual, threshold=cutoff)
    )
    return correct / len(probabilities)


def filter_valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = ("technical_score", "news_score", "regime_score", "actual_return_pct", "horizon_days")
    return [
        row
        for row in rows
        if all(row.get(key) is not None for key in required)
    ]


def chronological_split(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    ordered = sorted(rows, key=lambda row: row["timestamp"])
    split_index = max(1, int(len(ordered) * train_fraction))
    if split_index >= len(ordered):
        split_index = len(ordered) - 1
    return ordered[:split_index], ordered[split_index:]


def group_by_horizon(
    rows: list[dict[str, Any]],
    horizons: tuple[int, ...],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {horizon: [] for horizon in horizons}
    for row in rows:
        horizon = int(row["horizon_days"])
        if horizon in grouped:
            grouped[horizon].append(row)
    return grouped


def _metrics_for_rows(
    rows: list[dict[str, Any]],
    weights: FusionWeights,
    scales_by_horizon: dict[int, float],
    *,
    threshold: float | None = None,
) -> dict[str, float | None]:
    if not rows:
        return {"brier": None, "direction_accuracy": None, "sample_size": 0}

    probabilities: list[float] = []
    actuals: list[float] = []
    for row in rows:
        horizon = int(row["horizon_days"])
        scale = scales_by_horizon.get(horizon, 1.0)
        probabilities.append(_sigmoid(combined_score(row, weights) * scale))
        actuals.append(float(row["actual_return_pct"]))

    return {
        "brier": round(brier_score(probabilities, actuals), 4),
        "direction_accuracy": round(
            direction_accuracy(probabilities, actuals, threshold=threshold),
            4,
        ),
        "sample_size": len(rows),
    }


def _pick_scales_on_train(
    train_by_horizon: dict[int, list[dict[str, Any]]],
    weights: FusionWeights,
    scale_candidates: list[float],
) -> dict[int, float]:
    scales: dict[int, float] = {}
    for horizon, train_rows in train_by_horizon.items():
        if not train_rows:
            scales[horizon] = HORIZON_SCALES.get(horizon, 1.0)
            continue

        actuals = [float(row["actual_return_pct"]) for row in train_rows]
        best_scale = scale_candidates[0]
        best_brier = float("inf")
        for scale in scale_candidates:
            probs = probabilities_for_rows(train_rows, weights, scale)
            score = brier_score(probs, actuals)
            if score < best_brier:
                best_brier = score
                best_scale = scale
        scales[horizon] = best_scale
    return scales


def _weighted_validation_brier(
    val_by_horizon: dict[int, list[dict[str, Any]]],
    weights: FusionWeights,
    scales_by_horizon: dict[int, float],
) -> float:
    total_weight = 0
    weighted_sum = 0.0
    for horizon, val_rows in val_by_horizon.items():
        if not val_rows:
            continue
        actuals = [float(row["actual_return_pct"]) for row in val_rows]
        scale = scales_by_horizon.get(horizon, 1.0)
        probs = probabilities_for_rows(val_rows, weights, scale)
        score = brier_score(probs, actuals)
        weighted_sum += score * len(val_rows)
        total_weight += len(val_rows)
    if total_weight == 0:
        raise FusionOptimizationError("No validation rows available for scoring")
    return weighted_sum / total_weight


def _weights_dict(weights: FusionWeights) -> dict[str, float]:
    return {
        "technical": round(weights[0], 4),
        "news": round(weights[1], 4),
        "regime": round(weights[2], 4),
    }


def _scales_dict(scales: dict[int, float]) -> dict[str, float]:
    return {str(horizon): round(scale, 4) for horizon, scale in sorted(scales.items())}


def optimize_fusion(
    rows: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...],
    train_fraction: float = 0.7,
    min_samples: int = 30,
    weight_step: float = 0.05,
    scale_candidates: list[float] | None = None,
    baseline_weights: FusionWeights | None = None,
    baseline_scales: dict[int, float] | None = None,
) -> dict[str, Any]:
    valid_rows = filter_valid_rows(rows)
    scale_candidates = scale_candidates or grid_scale_candidates()
    baseline_weights = baseline_weights or DEFAULT_WEIGHTS
    baseline_scales = baseline_scales or dict(HORIZON_SCALES)

    train_rows, val_rows = chronological_split(valid_rows, train_fraction=train_fraction)
    train_by_horizon = group_by_horizon(train_rows, horizons)
    val_by_horizon = group_by_horizon(val_rows, horizons)

    sample_sizes = {
        "train": {str(h): len(train_by_horizon[h]) for h in horizons},
        "validation": {str(h): len(val_by_horizon[h]) for h in horizons},
    }

    for horizon in horizons:
        total = sample_sizes["train"][str(horizon)] + sample_sizes["validation"][str(horizon)]
        if total < min_samples:
            raise FusionOptimizationError(
                f"Horizon {horizon}d has only {total} evaluated rows "
                f"(need at least {min_samples}). "
                "Run tradingai-predict-eval after more predictions mature."
            )
        if sample_sizes["validation"][str(horizon)] == 0:
            raise FusionOptimizationError(
                f"Horizon {horizon}d has no validation rows after chronological split. "
                "Collect more evaluated predictions or lower --train-fraction."
            )

    best_weights = baseline_weights
    best_scales = dict(baseline_scales)
    best_validation_brier = _weighted_validation_brier(val_by_horizon, best_weights, best_scales)

    for candidate_weights in grid_weight_candidates(step=weight_step):
        candidate_scales = _pick_scales_on_train(train_by_horizon, candidate_weights, scale_candidates)
        validation_brier = _weighted_validation_brier(
            val_by_horizon,
            candidate_weights,
            candidate_scales,
        )
        if validation_brier < best_validation_brier:
            best_validation_brier = validation_brier
            best_weights = candidate_weights
            best_scales = candidate_scales

    val_flat = [row for rows in val_by_horizon.values() for row in rows]
    baseline_metrics = _metrics_for_rows(val_flat, baseline_weights, baseline_scales)
    tuned_metrics = _metrics_for_rows(val_flat, best_weights, best_scales)

    return {
        "tuned_at": datetime.now().astimezone().isoformat(),
        "metric": "brier",
        "train_fraction": train_fraction,
        "weights": _weights_dict(best_weights),
        "horizon_scales": _scales_dict(best_scales),
        "sample_sizes": sample_sizes,
        "baseline_validation": baseline_metrics,
        "tuned_validation": tuned_metrics,
    }
