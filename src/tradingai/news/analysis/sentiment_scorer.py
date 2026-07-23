from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, TypeVar

from tradingai.news.config import get_news_scoring_params
from tradingai.news.models import AnalystAction, NewsItem, SocialPost

T = TypeVar("T")

BULLISH_TERMS = {
    "beat",
    "growth",
    "strong",
    "upgrade",
    "raised",
    "outperform",
    "record",
    "surge",
    "bullish",
    "expansion",
    "demand",
    "profit",
}

BEARISH_TERMS = {
    "miss",
    "weak",
    "downgrade",
    "lawsuit",
    "decline",
    "loss",
    "risk",
    "bearish",
    "cut",
    "investigation",
    "shortage",
    "sanction",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def score_text(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.5
    bullish = len(tokens & BULLISH_TERMS)
    bearish = len(tokens & BEARISH_TERMS)
    total = bullish + bearish
    if total == 0:
        return 0.5
    return max(0.0, min(1.0, 0.5 + (bullish - bearish) / (2 * total)))


def is_published_as_of(
    published_at: datetime | None,
    *,
    now: datetime | None = None,
) -> bool:
    if published_at is None:
        return True
    reference = now or datetime.now(timezone.utc)
    return published_at.astimezone(timezone.utc) <= reference.astimezone(timezone.utc)


def filter_by_published_at(items: list[T], *, now: datetime | None = None) -> list[T]:
    return [item for item in items if is_published_as_of(item.published_at, now=now)]


def _recency_params(params: dict[str, Any] | None = None) -> dict[str, float]:
    scoring = params or get_news_scoring_params()
    recency = scoring.get("recency", {})
    return {
        "half_life_days": float(recency.get("half_life_days", 7.0)),
        "floor": float(recency.get("floor", 0.4)),
        "missing_date_weight": float(recency.get("missing_date_weight", 0.6)),
    }


def recency_weight(
    published_at: datetime | None,
    now: datetime,
    *,
    half_life_days: float | None = None,
    floor: float | None = None,
    missing_date_weight: float | None = None,
    params: dict[str, Any] | None = None,
) -> float:
    defaults = _recency_params(params)
    half_life = half_life_days if half_life_days is not None else defaults["half_life_days"]
    floor_value = floor if floor is not None else defaults["floor"]
    missing = (
        missing_date_weight
        if missing_date_weight is not None
        else defaults["missing_date_weight"]
    )
    if published_at is None:
        return missing
    age = now - published_at.astimezone(timezone.utc)
    age_days = max(0.0, age.total_seconds() / 86400.0)
    if half_life <= 0:
        return floor_value
    return max(floor_value, math.exp(-age_days / half_life))


def _parse_published_at(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def snapshot_item(
    channel: str,
    base_score: float,
    published_at: datetime | None = None,
    *,
    engagement: float = 1.0,
) -> dict[str, Any]:
    return {
        "channel": channel,
        "base_score": round(float(base_score), 4),
        "published_at": published_at.astimezone(timezone.utc).isoformat() if published_at else None,
        "engagement": round(float(engagement), 4),
    }


def _channel_aggregate(
    items: list[dict[str, Any]],
    now: datetime,
    params: dict[str, Any],
) -> float | None:
    if not items:
        return None
    weighted_scores: list[float] = []
    weights: list[float] = []
    for item in items:
        published_at = _parse_published_at(item.get("published_at"))
        weight = recency_weight(published_at, now, params=params)
        engagement = float(item.get("engagement", 1.0) or 1.0)
        weight *= engagement
        weighted_scores.append(float(item["base_score"]) * weight)
        weights.append(weight)
    if not weights:
        return None
    return sum(weighted_scores) / sum(weights)


def rescore_from_snapshot(
    snapshot: dict[str, Any] | None,
    params: dict[str, Any] | None = None,
    *,
    as_of: datetime | None = None,
) -> float:
    """Re-aggregate sentiment from a stored scoring snapshot."""
    scoring = params or get_news_scoring_params()
    source_weights = scoring.get("source_weights", {})
    now = as_of or datetime.now(timezone.utc)
    items = (snapshot or {}).get("items") or []
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        channel = str(item.get("channel", "")).strip().lower()
        if not channel or item.get("base_score") is None:
            continue
        by_channel.setdefault(channel, []).append(item)

    weighted_scores: list[tuple[float, float]] = []
    for channel, channel_items in by_channel.items():
        channel_score = _channel_aggregate(channel_items, now, scoring)
        if channel_score is None:
            continue
        weight = float(source_weights.get(channel, 1.0))
        weighted_scores.append((channel_score, weight))
    return aggregate_sentiment(weighted_scores)


def score_news_items(
    items: list[NewsItem],
    *,
    channel: str = "news",
    now: datetime | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[float | None, list[dict[str, Any]]]:
    if not items:
        return None, []
    reference = now or datetime.now(timezone.utc)
    scoring = params or get_news_scoring_params()
    snapshot_items: list[dict[str, Any]] = []
    weighted_scores: list[float] = []
    weights: list[float] = []
    for item in items:
        if not is_published_as_of(item.published_at, now=reference):
            continue
        text = " ".join(filter(None, [item.title, item.summary]))
        base = score_text(text)
        snapshot_items.append(snapshot_item(channel, base, item.published_at))
        weight = recency_weight(item.published_at, reference, params=scoring)
        weighted_scores.append(base * weight)
        weights.append(weight)
    if not weights:
        return None, snapshot_items
    return sum(weighted_scores) / sum(weights), snapshot_items


def score_analyst_actions(
    actions: list[AnalystAction],
    *,
    now: datetime | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[float | None, list[dict[str, Any]]]:
    if not actions:
        return None, []
    reference = now or datetime.now(timezone.utc)
    scoring = params or get_news_scoring_params()
    snapshot_items: list[dict[str, Any]] = []
    weighted_scores: list[float] = []
    weights: list[float] = []
    for action in actions:
        if not is_published_as_of(action.published_at, now=reference):
            continue
        action_text = action.action.lower()
        if "upgrade" in action_text or "outperform" in action_text:
            base = 0.8
        elif "downgrade" in action_text or "underperform" in action_text:
            base = 0.2
        elif "initiated" in action_text:
            base = 0.65
        else:
            base = 0.5
        snapshot_items.append(snapshot_item("analyst", base, action.published_at))
        weight = recency_weight(action.published_at, reference, params=scoring)
        weighted_scores.append(base * weight)
        weights.append(weight)
    if not weights:
        return None, snapshot_items
    return sum(weighted_scores) / sum(weights), snapshot_items


def score_social_posts(
    posts: list[SocialPost],
    *,
    now: datetime | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[float | None, list[dict[str, Any]]]:
    if not posts:
        return None, []
    reference = now or datetime.now(timezone.utc)
    scoring = params or get_news_scoring_params()
    snapshot_items: list[dict[str, Any]] = []
    weighted_scores: list[float] = []
    weights: list[float] = []
    for post in posts:
        if not is_published_as_of(post.published_at, now=reference):
            continue
        text = " ".join(filter(None, [post.title, post.summary]))
        base = score_text(text)
        engagement = 1.0
        if post.engagement_score is not None:
            engagement = min(1.5, 1.0 + post.engagement_score / 100.0)
        snapshot_items.append(
            snapshot_item("social", base, post.published_at, engagement=engagement)
        )
        recency = recency_weight(post.published_at, reference, params=scoring)
        weight = recency * engagement
        weighted_scores.append(base * weight)
        weights.append(weight)
    if not weights:
        return None, snapshot_items
    return sum(weighted_scores) / sum(weights), snapshot_items


def aggregate_sentiment(scores: list[tuple[float, float]]) -> float:
    if not scores:
        return 0.5
    numerator = sum(score * weight for score, weight in scores)
    denominator = sum(weight for _, weight in scores)
    if denominator == 0:
        return 0.5
    return max(0.0, min(1.0, numerator / denominator))


def compute_confidence(
    source_count: int,
    item_count: int,
    sentiment_scores: list[float],
) -> float:
    source_component = min(1.0, source_count / 3.0)
    coverage_component = min(1.0, item_count / 10.0)
    if len(sentiment_scores) >= 2:
        mean = sum(sentiment_scores) / len(sentiment_scores)
        variance = sum((score - mean) ** 2 for score in sentiment_scores) / len(
            sentiment_scores
        )
        agreement_component = max(0.0, 1.0 - math.sqrt(variance) * 2)
    else:
        agreement_component = 0.4
    confidence = (
        0.45 * source_component + 0.35 * coverage_component + 0.20 * agreement_component
    )
    return max(0.0, min(1.0, confidence))
