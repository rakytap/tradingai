from __future__ import annotations

from typing import Any

DEFAULT_NEIGHBOR_LIMIT = 5
DEFAULT_MAX_AGE_MINUTES = 120


def neighbor_directional_from_snapshot(payload: dict | None) -> float | None:
    """Map a cached sentiment snapshot to a directional score in [-1, 1]."""
    if not isinstance(payload, dict):
        return None
    event_score = payload.get("event_score")
    if event_score is not None:
        try:
            return max(-1.0, min(1.0, float(event_score)))
        except (TypeError, ValueError):
            pass
    sentiment = payload.get("sentiment")
    if sentiment is None:
        return None
    try:
        return max(-1.0, min(1.0, (float(sentiment) - 0.5) * 2.0))
    except (TypeError, ValueError):
        return None


def blend_l3_news_score(
    own: float,
    candidates: list[dict],
    neighbor_scores: dict[str, float],
    alpha: float,
) -> tuple[float, dict[str, Any]]:
    """Blend own news score with weighted neighbor scores.

    score = (1-α)*own + α*Σ(w_i*s_i)/Σw_i
    """
    own_clipped = max(-1.0, min(1.0, float(own)))
    alpha_clipped = max(0.0, min(1.0, float(alpha)))
    used: list[dict[str, Any]] = []
    weight_sum = 0.0
    weighted = 0.0

    for candidate in candidates:
        ticker = str(candidate.get("ticker") or "").upper()
        if not ticker or ticker not in neighbor_scores:
            continue
        try:
            weight = float(candidate.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        score = float(neighbor_scores[ticker])
        weight_sum += weight
        weighted += weight * score
        used.append({"ticker": ticker, "weight": round(weight, 4), "score": round(score, 4)})

    if not used or weight_sum <= 0 or alpha_clipped <= 0:
        adjusted = own_clipped
        neighbor_avg = None
    else:
        neighbor_avg = weighted / weight_sum
        adjusted = (1.0 - alpha_clipped) * own_clipped + alpha_clipped * neighbor_avg

    adjusted = max(-1.0, min(1.0, adjusted))
    meta = {
        "alpha": round(alpha_clipped, 4),
        "own": round(own_clipped, 4),
        "neighbor_avg": None if neighbor_avg is None else round(neighbor_avg, 4),
        "used": used,
        "adjusted": round(adjusted, 4),
    }
    return adjusted, meta


def load_neighbor_news_scores(
    candidates: list[dict],
    *,
    max_age_minutes: int = DEFAULT_MAX_AGE_MINUTES,
    limit: int = DEFAULT_NEIGHBOR_LIMIT,
) -> dict[str, float]:
    """Load directional scores from cached sentiment snapshots for top candidates."""
    from tradingai.data.repositories.news_events import get_latest_sentiment_snapshot

    ranked = sorted(
        (c for c in candidates if isinstance(c, dict) and c.get("ticker")),
        key=lambda c: float(c.get("weight") or 0.0),
        reverse=True,
    )[: max(0, limit)]

    scores: dict[str, float] = {}
    for candidate in ranked:
        ticker = str(candidate["ticker"]).upper()
        if ticker in scores:
            continue
        try:
            snapshot = get_latest_sentiment_snapshot(
                ticker, max_age_minutes=max_age_minutes
            )
        except Exception:
            continue
        directional = neighbor_directional_from_snapshot(snapshot)
        if directional is not None:
            scores[ticker] = directional
    return scores
