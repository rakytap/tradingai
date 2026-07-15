from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import TypeVar

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


def _recency_weight(published_at: datetime | None, now: datetime) -> float:
    if published_at is None:
        return 0.6
    age = now - published_at.astimezone(timezone.utc)
    if age <= timedelta(days=2):
        return 1.0
    if age <= timedelta(days=7):
        return 0.85
    if age <= timedelta(days=30):
        return 0.65
    return 0.4


def score_news_items(items: list[NewsItem]) -> float | None:
    if not items:
        return None
    now = datetime.now(timezone.utc)
    weighted_scores: list[float] = []
    weights: list[float] = []
    for item in items:
        if not is_published_as_of(item.published_at, now=now):
            continue
        text = " ".join(filter(None, [item.title, item.summary]))
        weight = _recency_weight(item.published_at, now)
        weighted_scores.append(score_text(text) * weight)
        weights.append(weight)
    if not weights:
        return None
    return sum(weighted_scores) / sum(weights)


def score_analyst_actions(actions: list[AnalystAction]) -> float | None:
    if not actions:
        return None
    now = datetime.now(timezone.utc)
    scores: list[float] = []
    for action in actions:
        if not is_published_as_of(action.published_at, now=now):
            continue
        action_text = action.action.lower()
        if "upgrade" in action_text or "outperform" in action_text:
            scores.append(0.8)
        elif "downgrade" in action_text or "underperform" in action_text:
            scores.append(0.2)
        elif "initiated" in action_text:
            scores.append(0.65)
        else:
            scores.append(0.5)
    if not scores:
        return None
    return sum(scores) / len(scores)


def score_social_posts(posts: list[SocialPost]) -> float | None:
    if not posts:
        return None
    now = datetime.now(timezone.utc)
    weighted_scores: list[float] = []
    weights: list[float] = []
    for post in posts:
        if not is_published_as_of(post.published_at, now=now):
            continue
        text = " ".join(filter(None, [post.title, post.summary]))
        base = score_text(text)
        recency = _recency_weight(post.published_at, now)
        engagement = 1.0
        if post.engagement_score is not None:
            engagement = min(1.5, 1.0 + post.engagement_score / 100.0)
        weight = recency * engagement
        weighted_scores.append(base * weight)
        weights.append(weight)
    if not weights:
        return None
    return sum(weighted_scores) / sum(weights)


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
