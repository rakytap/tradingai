from __future__ import annotations

from tradingai.analysis.ml.config import get_direction_threshold


def predicted_direction_up(
    probability_up: float,
    *,
    threshold: float | None = None,
) -> bool:
    cutoff = get_direction_threshold() if threshold is None else threshold
    return float(probability_up) >= cutoff


def actual_direction_up(actual_return_pct: float) -> bool:
    return float(actual_return_pct) > 0


def direction_is_correct(
    probability_up: float,
    actual_return_pct: float,
    *,
    threshold: float | None = None,
) -> bool:
    return predicted_direction_up(probability_up, threshold=threshold) == actual_direction_up(
        actual_return_pct
    )
