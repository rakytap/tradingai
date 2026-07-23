from __future__ import annotations

from datetime import datetime, timezone

from tradingai.news.analysis.catalyst_extractor import extract_catalysts, extract_risk_factors
from tradingai.news.analysis.sentiment_scorer import (
    aggregate_sentiment,
    compute_confidence,
    filter_by_published_at,
    recency_weight,
    score_analyst_actions,
    score_news_items,
    score_social_posts,
    score_text,
    snapshot_item,
)
from tradingai.news.config import get_news_scoring_params
from tradingai.news.models import (
    AnalystAction,
    EarningsMaterial,
    FilingItem,
    NewsItem,
    SentimentReport,
    SocialPost,
)


def _social_as_news(posts: list[SocialPost]) -> list[NewsItem]:
    return [
        NewsItem(
            title=post.title,
            summary=post.summary,
            url=post.url,
            source=post.source,
            published_at=post.published_at,
            provider=post.provider,
        )
        for post in posts
    ]


def synthesize_sentiment_report(
    ticker: str,
    news: list[NewsItem],
    press_releases: list[NewsItem],
    filings: list[FilingItem],
    analyst_actions: list[AnalystAction],
    earnings: EarningsMaterial | None = None,
    social_posts: list[SocialPost] | None = None,
    sources_used: list[str] | None = None,
) -> SentimentReport:
    now = datetime.now(timezone.utc)
    scoring = get_news_scoring_params()
    source_weights = scoring["source_weights"]

    news = filter_by_published_at(news, now=now)
    press_releases = filter_by_published_at(press_releases, now=now)
    social_posts = filter_by_published_at(social_posts or [], now=now)
    analyst_actions = filter_by_published_at(analyst_actions, now=now)

    earnings_text = earnings.press_release if earnings else None
    transcript_text = earnings.transcript if earnings else None
    social_news = _social_as_news(social_posts)

    catalysts = extract_catalysts(
        news + social_news,
        press_releases,
        filings,
        analyst_actions,
        earnings_text,
        transcript_text,
    )
    risk_factors = extract_risk_factors(
        news + social_news,
        press_releases,
        filings,
        earnings_text,
        transcript_text,
    )

    weighted_scores: list[tuple[float, float]] = []
    raw_scores: list[float] = []
    snapshot_items: list[dict] = []

    news_score, news_items = score_news_items(news, channel="news", now=now, params=scoring)
    snapshot_items.extend(news_items)
    if news_score is not None:
        weighted_scores.append((news_score, float(source_weights["news"])))
        raw_scores.append(news_score)

    press_score, press_items = score_news_items(
        press_releases, channel="press", now=now, params=scoring
    )
    snapshot_items.extend(press_items)
    if press_score is not None:
        weighted_scores.append((press_score, float(source_weights["press"])))
        raw_scores.append(press_score)

    analyst_score, analyst_items = score_analyst_actions(
        analyst_actions, now=now, params=scoring
    )
    snapshot_items.extend(analyst_items)
    if analyst_score is not None:
        weighted_scores.append((analyst_score, float(source_weights["analyst"])))
        raw_scores.append(analyst_score)

    filing_snapshot: list[dict] = []
    filing_weighted: list[float] = []
    filing_weights: list[float] = []
    for filing in filings:
        if not filing.excerpt:
            continue
        filing_score = score_text(filing.excerpt)
        published = filing.filed_at
        filing_snapshot.append(snapshot_item("filings", filing_score, published))
        weight = recency_weight(published, now, params=scoring)
        filing_weighted.append(filing_score * weight)
        filing_weights.append(weight)
    snapshot_items.extend(filing_snapshot)
    if filing_weights:
        channel_score = sum(filing_weighted) / sum(filing_weights)
        weighted_scores.append((channel_score, float(source_weights["filings"])))
        raw_scores.append(channel_score)

    if earnings_text:
        earnings_score = score_text(earnings_text)
        published = earnings.filed_at if earnings else None
        snapshot_items.append(snapshot_item("earnings", earnings_score, published))
        # Single-item channel: recency applied in rescore; live uses same via weight 1.0 item.
        weighted_scores.append((earnings_score, float(source_weights["earnings"])))
        raw_scores.append(earnings_score)

    if transcript_text:
        transcript_score = score_text(transcript_text)
        published = earnings.filed_at if earnings else None
        snapshot_items.append(snapshot_item("transcript", transcript_score, published))
        weighted_scores.append((transcript_score, float(source_weights["transcript"])))
        raw_scores.append(transcript_score)

    social_score, social_items = score_social_posts(social_posts, now=now, params=scoring)
    snapshot_items.extend(social_items)
    if social_score is not None:
        weighted_scores.append((social_score, float(source_weights["social"])))
        raw_scores.append(social_score)

    sentiment = aggregate_sentiment(weighted_scores)
    confidence = compute_confidence(
        source_count=len(set(sources_used or [])),
        item_count=len(news)
        + len(press_releases)
        + len(filings)
        + len(analyst_actions)
        + len(social_posts),
        sentiment_scores=raw_scores,
    )

    return SentimentReport(
        ticker=ticker.upper(),
        sentiment=sentiment,
        catalysts=catalysts,
        risk_factors=risk_factors,
        confidence=confidence,
        sources_used=sources_used or [],
        scoring_snapshot={"items": snapshot_items},
    )
