from datetime import datetime, timedelta, timezone

from tradingai.news.analysis.catalyst_extractor import extract_catalysts, extract_risk_factors
from tradingai.news.analysis.sentiment_scorer import (
    aggregate_sentiment,
    compute_confidence,
    filter_by_published_at,
    is_published_as_of,
    score_news_items,
    score_text,
)
from tradingai.news.analysis.synthesizer import synthesize_sentiment_report
from tradingai.news.models import AnalystAction, FilingItem, NewsItem, SentimentReport

FIXED_NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


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


def test_is_published_as_of_keeps_none_past_and_now():
    assert is_published_as_of(None, now=FIXED_NOW)
    assert is_published_as_of(FIXED_NOW - timedelta(days=1), now=FIXED_NOW)
    assert is_published_as_of(FIXED_NOW, now=FIXED_NOW)


def test_is_published_as_of_rejects_future():
    assert not is_published_as_of(FIXED_NOW + timedelta(hours=1), now=FIXED_NOW)


def test_filter_by_published_at_excludes_future_items():
    past = NewsItem(
        title="Past headline",
        source="Test",
        provider="yfinance",
        published_at=FIXED_NOW - timedelta(days=1),
    )
    future = NewsItem(
        title="Future headline",
        source="Test",
        provider="yfinance",
        published_at=FIXED_NOW + timedelta(days=1),
    )
    filtered = filter_by_published_at([past, future], now=FIXED_NOW)
    assert filtered == [past]


def test_score_news_items_excludes_future_published_at():
    past = NewsItem(
        title="Weak demand and downgrade after lawsuit risk",
        source="Test",
        provider="yfinance",
        published_at=FIXED_NOW - timedelta(days=1),
    )
    future = NewsItem(
        title="Strong revenue growth and earnings beat estimates",
        source="Test",
        provider="yfinance",
        published_at=FIXED_NOW + timedelta(days=1),
    )
    score = score_news_items([past, future])
    assert score is not None
    assert score < 0.5


def test_synthesize_sentiment_report_ignores_future_headlines(monkeypatch):
    import tradingai.news.analysis.synthesizer as synthesizer_module

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_NOW

    monkeypatch.setattr(synthesizer_module, "datetime", FixedDatetime)

    past = NewsItem(
        title="Weak demand and downgrade after lawsuit risk",
        source="Test",
        provider="yfinance",
        published_at=FIXED_NOW - timedelta(days=1),
    )
    future = NewsItem(
        title="Strong revenue growth and earnings beat estimates",
        source="Test",
        provider="yfinance",
        published_at=FIXED_NOW + timedelta(days=1),
    )
    report = synthesize_sentiment_report(
        ticker="NVDA",
        news=[past, future],
        press_releases=[],
        filings=[],
        analyst_actions=[],
        sources_used=["yfinance"],
    )
    assert report.sentiment < 0.5


def test_synthesize_sentiment_report_excludes_future_catalysts(monkeypatch):
    import tradingai.news.analysis.synthesizer as synthesizer_module

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_NOW

    monkeypatch.setattr(synthesizer_module, "datetime", FixedDatetime)

    future_only = NewsItem(
        title="Company beat estimates on AI infrastructure demand",
        summary="Revenue growth accelerated in data centers.",
        source="Test",
        provider="yfinance",
        published_at=FIXED_NOW + timedelta(days=1),
    )
    report = synthesize_sentiment_report(
        ticker="NVDA",
        news=[future_only],
        press_releases=[],
        filings=[],
        analyst_actions=[],
        sources_used=["yfinance"],
    )
    assert not any("Earnings" in catalyst for catalyst in report.catalysts)
