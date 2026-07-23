from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from tradingai.news.analysis.sentiment_scorer import rescore_from_snapshot
from tradingai.news.config import (
    DEFAULT_RECENCY,
    DEFAULT_SOURCE_WEIGHTS,
    default_news_scoring_params,
)
from tradingai.predict.fusion.optimizer import (
    _metrics_for_rows,
    chronological_split,
)
from tradingai.predict.fusion.scorer import _sigmoid

HALF_LIFE_CANDIDATES = (2.0, 5.0, 7.0, 10.0, 14.0, 21.0)
SOCIAL_WEIGHT_CANDIDATES = (0.2, 0.35, 0.45, 0.6, 0.8)

SOURCE_WEIGHT_TEMPLATES: tuple[dict[str, float], ...] = (
    dict(DEFAULT_SOURCE_WEIGHTS),
    {
        **DEFAULT_SOURCE_WEIGHTS,
        "news": 0.9,
        "press": 1.0,
        "analyst": 1.6,
        "earnings": 1.5,
        "social": DEFAULT_SOURCE_WEIGHTS["social"],
    },
    {
        **DEFAULT_SOURCE_WEIGHTS,
        "news": 1.3,
        "press": 1.3,
        "analyst": 1.0,
        "earnings": 1.1,
        "social": DEFAULT_SOURCE_WEIGHTS["social"],
    },
)


class NewsScoringOptimizationError(ValueError):
    """Raised when news scoring cannot be tuned (e.g. missing snapshots)."""


def sentiment_to_directional(sentiment: float) -> float:
    return max(-1.0, min(1.0, (float(sentiment) - 0.5) * 2.0))


def _parse_agent_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("agent_payload")
    if payload is None:
        return {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def extract_scoring_snapshot(row: dict[str, Any]) -> dict[str, Any] | None:
    payload = _parse_agent_payload(row)
    news = payload.get("news")
    if isinstance(news, dict):
        snapshot = news.get("scoring_snapshot")
        if isinstance(snapshot, dict) and snapshot.get("items"):
            return snapshot
    snapshot = payload.get("scoring_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("items"):
        return snapshot
    return None


def rows_with_scoring_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if extract_scoring_snapshot(row) is not None]


def _as_of_for_row(row: dict[str, Any]) -> datetime:
    timestamp = row.get("timestamp")
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def build_scoring_params(
    *,
    source_weights: dict[str, float],
    half_life_days: float,
    floor: float = DEFAULT_RECENCY["floor"],
    missing_date_weight: float = DEFAULT_RECENCY["missing_date_weight"],
) -> dict[str, Any]:
    return {
        "source_weights": dict(source_weights),
        "recency": {
            "half_life_days": float(half_life_days),
            "floor": float(floor),
            "missing_date_weight": float(missing_date_weight),
        },
    }


def resimulate_news_scores(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return copies of rows with news_score replaced from rescored snapshots."""
    updated: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        snapshot = extract_scoring_snapshot(row)
        if snapshot is not None:
            sentiment = rescore_from_snapshot(
                snapshot,
                params,
                as_of=_as_of_for_row(row),
            )
            copy["news_score"] = round(sentiment_to_directional(sentiment), 4)
        updated.append(copy)
    return updated


def _combined_probability(
    row: dict[str, Any],
    *,
    news_score: float,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> float:
    w_technical, w_news, w_regime = fusion_weights
    technical = float(row["technical_score"])
    regime = float(row["regime_score"])
    combined = w_technical * technical + w_news * news_score + w_regime * regime
    scale = horizon_scales.get(int(row["horizon_days"]), 1.0)
    return _sigmoid(combined * scale)


def _brier_for_params(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> float:
    if not rows:
        raise NewsScoringOptimizationError("No rows available for news Brier scoring")
    total = 0.0
    for row in rows:
        snapshot = extract_scoring_snapshot(row)
        if snapshot is None:
            continue
        sentiment = rescore_from_snapshot(snapshot, params, as_of=_as_of_for_row(row))
        news_score = sentiment_to_directional(sentiment)
        prob = _combined_probability(
            row,
            news_score=news_score,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
        )
        actual_up = 1.0 if float(row["actual_return_pct"]) > 0 else 0.0
        total += (prob - actual_up) ** 2
    return total / len(rows)


def _metrics_with_news_params(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> dict[str, float | None]:
    simulated = resimulate_news_scores(rows, params)
    return _metrics_for_rows(simulated, fusion_weights, horizon_scales)


def grid_news_scoring_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for template in SOURCE_WEIGHT_TEMPLATES:
        for social in SOCIAL_WEIGHT_CANDIDATES:
            weights = dict(template)
            weights["social"] = float(social)
            for half_life in HALF_LIFE_CANDIDATES:
                candidates.append(
                    build_scoring_params(
                        source_weights=weights,
                        half_life_days=half_life,
                    )
                )
    return candidates


def optimize_news_scoring(
    rows: list[dict[str, Any]],
    *,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
    train_fraction: float = 0.7,
    min_samples: int = 30,
) -> dict[str, Any]:
    eligible = rows_with_scoring_snapshots(rows)
    if len(eligible) < min_samples:
        raise NewsScoringOptimizationError(
            f"Only {len(eligible)} evaluated rows have scoring_snapshot "
            f"(need at least {min_samples}). "
            "Generate new predictions after deploying snapshot support, "
            "run tradingai-predict-eval, then re-run tradingai-tune-fusion."
        )

    required = ("technical_score", "regime_score", "actual_return_pct", "horizon_days")
    eligible = [
        row
        for row in eligible
        if all(row.get(key) is not None for key in required)
    ]
    if len(eligible) < min_samples:
        raise NewsScoringOptimizationError(
            f"Only {len(eligible)} snapshot rows have complete fusion component scores "
            f"(need at least {min_samples})."
        )

    train_rows, val_rows = chronological_split(eligible, train_fraction=train_fraction)
    if not val_rows:
        raise NewsScoringOptimizationError(
            "No validation rows after chronological split for news scoring. "
            "Collect more evaluated predictions or lower --train-fraction."
        )

    baseline_params = default_news_scoring_params()
    best_params = deepcopy(baseline_params)
    best_brier = _brier_for_params(
        val_rows,
        best_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )

    for candidate in grid_news_scoring_candidates():
        val_brier = _brier_for_params(
            val_rows,
            candidate,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
        )
        if val_brier < best_brier:
            best_brier = val_brier
            best_params = candidate

    baseline_metrics = _metrics_with_news_params(
        val_rows,
        baseline_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    tuned_metrics = _metrics_with_news_params(
        val_rows,
        best_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )

    return {
        "tuned_at": datetime.now().astimezone().isoformat(),
        "metric": "brier",
        "train_fraction": train_fraction,
        "source_weights": {
            key: round(float(value), 4)
            for key, value in best_params["source_weights"].items()
        },
        "recency": {
            key: round(float(value), 4) for key, value in best_params["recency"].items()
        },
        "sample_sizes": {
            "eligible": len(eligible),
            "train": len(train_rows),
            "validation": len(val_rows),
        },
        "baseline_validation": baseline_metrics,
        "tuned_validation": tuned_metrics,
    }
