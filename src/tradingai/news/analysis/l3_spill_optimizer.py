from __future__ import annotations

from datetime import datetime
from typing import Any

from tradingai.news.analysis.event_graph_optimizer import resolve_news_score_for_row
from tradingai.news.analysis.l3_spillover import (
    apply_l3_spill_to_score,
    extract_l3_spill_inputs,
    rows_with_l3_spill_inputs,
)
from tradingai.news.analysis.scoring_optimizer import _combined_probability
from tradingai.news.config import DEFAULT_L3_NEWS_SPILL_ALPHA
from tradingai.predict.fusion.optimizer import _metrics_for_rows, chronological_split

ALPHA_CANDIDATES = (0.0, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5)


class L3SpillOptimizationError(ValueError):
    """Raised when L3 spill alpha cannot be tuned."""


def _own_score_for_row(
    row: dict[str, Any],
    inputs: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    event_graph_params: dict[str, Any] | None,
) -> float:
    resolved = resolve_news_score_for_row(
        row,
        news_params=news_params,
        event_graph_params=event_graph_params,
    )
    if resolved is not None:
        return float(resolved)
    return float(inputs["own"])


def _news_score_with_alpha(
    row: dict[str, Any],
    alpha: float,
    *,
    news_params: dict[str, Any] | None,
    event_graph_params: dict[str, Any] | None,
) -> float | None:
    inputs = extract_l3_spill_inputs(row)
    if inputs is None:
        return None
    own = _own_score_for_row(
        row,
        inputs,
        news_params=news_params,
        event_graph_params=event_graph_params,
    )
    return apply_l3_spill_to_score(own, inputs, alpha)


def _brier_for_alpha(
    rows: list[dict[str, Any]],
    alpha: float,
    *,
    news_params: dict[str, Any] | None,
    event_graph_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> float:
    if not rows:
        raise L3SpillOptimizationError("No rows available for L3 spill Brier scoring")
    total = 0.0
    count = 0
    for row in rows:
        news_score = _news_score_with_alpha(
            row,
            alpha,
            news_params=news_params,
            event_graph_params=event_graph_params,
        )
        if news_score is None:
            continue
        prob = _combined_probability(
            row,
            news_score=news_score,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
        )
        actual_up = 1.0 if float(row["actual_return_pct"]) > 0 else 0.0
        total += (prob - actual_up) ** 2
        count += 1
    if count == 0:
        raise L3SpillOptimizationError("No L3 spill rows scored for Brier")
    return total / count


def _metrics_with_alpha(
    rows: list[dict[str, Any]],
    alpha: float,
    *,
    news_params: dict[str, Any] | None,
    event_graph_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> dict[str, float | None]:
    simulated: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        news_score = _news_score_with_alpha(
            row,
            alpha,
            news_params=news_params,
            event_graph_params=event_graph_params,
        )
        if news_score is not None:
            copy["news_score"] = news_score
        simulated.append(copy)
    return _metrics_for_rows(simulated, fusion_weights, horizon_scales)


def optimize_l3_spill_alpha(
    rows: list[dict[str, Any]],
    *,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
    news_params: dict[str, Any] | None = None,
    event_graph_params: dict[str, Any] | None = None,
    train_fraction: float = 0.7,
    min_samples: int = 30,
) -> dict[str, Any]:
    eligible = rows_with_l3_spill_inputs(rows)
    if len(eligible) < min_samples:
        raise L3SpillOptimizationError(
            f"Only {len(eligible)} evaluated rows have l3_news_spill inputs "
            f"(need at least {min_samples}). "
            "Generate predictions with TRADINGAI_L3_NEWS_SPILL=1 after neighbor "
            "snapshots exist, run tradingai-predict-eval, then re-run tradingai-tune-fusion."
        )

    required = ("technical_score", "regime_score", "actual_return_pct", "horizon_days")
    eligible = [
        row
        for row in eligible
        if all(row.get(key) is not None for key in required)
    ]
    if len(eligible) < min_samples:
        raise L3SpillOptimizationError(
            f"Only {len(eligible)} L3 spill rows have complete fusion component scores "
            f"(need at least {min_samples})."
        )

    train_rows, val_rows = chronological_split(eligible, train_fraction=train_fraction)
    if not val_rows:
        raise L3SpillOptimizationError(
            "No validation rows after chronological split for L3 spill alpha. "
            "Collect more evaluated predictions or lower --train-fraction."
        )

    best_alpha = float(DEFAULT_L3_NEWS_SPILL_ALPHA)
    best_brier = _brier_for_alpha(
        val_rows,
        best_alpha,
        news_params=news_params,
        event_graph_params=event_graph_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )

    for alpha in ALPHA_CANDIDATES:
        val_brier = _brier_for_alpha(
            val_rows,
            alpha,
            news_params=news_params,
            event_graph_params=event_graph_params,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
        )
        if val_brier < best_brier:
            best_brier = val_brier
            best_alpha = float(alpha)

    baseline_metrics = _metrics_with_alpha(
        val_rows,
        DEFAULT_L3_NEWS_SPILL_ALPHA,
        news_params=news_params,
        event_graph_params=event_graph_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    tuned_metrics = _metrics_with_alpha(
        val_rows,
        best_alpha,
        news_params=news_params,
        event_graph_params=event_graph_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )

    return {
        "tuned_at": datetime.now().astimezone().isoformat(),
        "metric": "brier",
        "train_fraction": train_fraction,
        "l3_news_spill_alpha": round(best_alpha, 4),
        "sample_sizes": {
            "eligible": len(eligible),
            "train": len(train_rows),
            "validation": len(val_rows),
        },
        "baseline_validation": baseline_metrics,
        "tuned_validation": tuned_metrics,
    }
