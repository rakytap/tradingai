from __future__ import annotations

from tradingai.news.analysis.catalyst_extractor import extract_catalysts, extract_risk_factors
from tradingai.news.analysis.sentiment_scorer import (
    aggregate_sentiment,
    compute_confidence,
    score_analyst_actions,
    score_news_items,
    score_social_posts,
    score_text,
)
from tradingai.news.config import get_social_weight
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
    earnings_text = earnings.press_release if earnings else None
    transcript_text = earnings.transcript if earnings else None
    social_news = _social_as_news(social_posts or [])

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

    news_score = score_news_items(news)
    if news_score is not None:
        weighted_scores.append((news_score, 1.0))
        raw_scores.append(news_score)

    press_score = score_news_items(press_releases)
    if press_score is not None:
        weighted_scores.append((press_score, 1.1))
        raw_scores.append(press_score)

    analyst_score = score_analyst_actions(analyst_actions)
    if analyst_score is not None:
        weighted_scores.append((analyst_score, 1.4))
        raw_scores.append(analyst_score)

    for filing in filings:
        if filing.excerpt:
            filing_score = score_text(filing.excerpt)
            weighted_scores.append((filing_score, 0.9))
            raw_scores.append(filing_score)

    if earnings_text:
        earnings_score = score_text(earnings_text)
        weighted_scores.append((earnings_score, 1.3))
        raw_scores.append(earnings_score)

    if transcript_text:
        transcript_score = score_text(transcript_text)
        weighted_scores.append((transcript_score, 1.0))
        raw_scores.append(transcript_score)

    social_score = score_social_posts(social_posts or [])
    if social_score is not None:
        social_weight = get_social_weight()
        weighted_scores.append((social_score, social_weight))
        raw_scores.append(social_score)

    sentiment = aggregate_sentiment(weighted_scores)
    confidence = compute_confidence(
        source_count=len(set(sources_used or [])),
        item_count=len(news)
        + len(press_releases)
        + len(filings)
        + len(analyst_actions)
        + len(social_posts or []),
        sentiment_scores=raw_scores,
    )

    return SentimentReport(
        ticker=ticker.upper(),
        sentiment=sentiment,
        catalysts=catalysts,
        risk_factors=risk_factors,
        confidence=confidence,
        sources_used=sources_used or [],
    )
