from datetime import datetime, timezone

from tradingai.news.analysis.catalyst_extractor import extract_catalysts, extract_risk_factors
from tradingai.news.analysis.sentiment_scorer import (
    aggregate_sentiment,
    compute_confidence,
    score_text,
)
from tradingai.news.analysis.synthesizer import synthesize_sentiment_report
from tradingai.news.models import AnalystAction, FilingItem, NewsItem, SentimentReport


def test_score_text_bullish():
    assert score_text("Strong revenue growth and earnings beat estimates") > 0.5


def test_score_text_bearish():
    assert score_text("Weak demand and downgrade after lawsuit risk") < 0.5


def test_extract_catalysts_from_headlines():
    news = [
        NewsItem(
            title="Company beat estimates on AI infrastructure demand",
            summary="Revenue growth accelerated in data centers.",
            source="Test",
            provider="yfinance",
        )
    ]
    catalysts = extract_catalysts(news, [], [], [])
    assert any("Earnings" in catalyst for catalyst in catalysts)


def test_extract_risk_factors_from_text():
    filings = [
        FilingItem(
            form="10-K",
            excerpt="The company faces valuation concerns and regulatory risk.",
            provider="sec_edgar",
        )
    ]
    risks = extract_risk_factors([], [], filings)
    assert "valuation" in risks
    assert "regulatory" in risks


def test_synthesize_sentiment_report_schema():
    news = [
        NewsItem(
            title="Strong product launch drives demand",
            summary="Analyst upgrade after rollout.",
            source="Test",
            provider="yfinance",
            published_at=datetime.now(timezone.utc),
        )
    ]
    actions = [
        AnalystAction(
            firm="Example Broker",
            action="upgrade",
            to_grade="Buy",
            provider="yfinance",
        )
    ]
    report = synthesize_sentiment_report(
        ticker="NVDA",
        news=news,
        press_releases=[],
        filings=[],
        analyst_actions=actions,
        sources_used=["yfinance"],
    )
    assert isinstance(report, SentimentReport)
    payload = report.to_public_dict()
    assert payload["ticker"] == "NVDA"
    assert 0.0 <= payload["sentiment"] <= 1.0
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["catalysts"], list)
    assert isinstance(payload["risk_factors"], list)


def test_aggregate_sentiment_and_confidence_bounds():
    sentiment = aggregate_sentiment([(0.8, 1.0), (0.7, 1.0)])
    confidence = compute_confidence(source_count=2, item_count=8, sentiment_scores=[0.8, 0.7])
    assert 0.0 <= sentiment <= 1.0
    assert 0.0 <= confidence <= 1.0
