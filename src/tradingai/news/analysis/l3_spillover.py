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


def _parse_agent_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("agent_payload")
    if payload is None:
        return {}
    if isinstance(payload, str):
        import json

        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def extract_l3_spill_inputs(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return stored L3 spill inputs `{own, used}` from prediction agent_payload."""
    payload = _parse_agent_payload(row)
    news = payload.get("news")
    spill = None
    if isinstance(news, dict):
        spill = news.get("l3_news_spill")
    if spill is None:
        spill = payload.get("l3_news_spill")
    if not isinstance(spill, dict):
        return None
    used = spill.get("used")
    if not isinstance(used, list) or not used:
        return None
    own = spill.get("own")
    if own is None:
        return None
    try:
        own_f = float(own)
    except (TypeError, ValueError):
        return None
    cleaned: list[dict[str, Any]] = []
    for item in used:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").upper()
        if not ticker:
            continue
        try:
            weight = float(item.get("weight", 0.0))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        cleaned.append({"ticker": ticker, "weight": weight, "score": score})
    if not cleaned:
        return None
    return {"own": own_f, "used": cleaned}


def rows_with_l3_spill_inputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if extract_l3_spill_inputs(row) is not None]


def apply_l3_spill_to_score(
    own: float,
    inputs: dict[str, Any],
    alpha: float,
) -> float:
    """Replay L3 spill blend using frozen neighbor scores from `inputs['used']`."""
    used = inputs.get("used") or []
    candidates = [{"ticker": u["ticker"], "weight": u["weight"]} for u in used]
    neighbor_scores = {str(u["ticker"]).upper(): float(u["score"]) for u in used}
    adjusted, _meta = blend_l3_news_score(own, candidates, neighbor_scores, alpha)
    return adjusted
