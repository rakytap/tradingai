from tradingai.news.analysis.catalyst_extractor import extract_catalysts, extract_risk_factors
from tradingai.news.models import AnalystAction, NewsItem


def test_analyst_upgrade_catalyst():
    actions = [
        AnalystAction(
            firm="Goldman Sachs",
            action="upgrade",
            from_grade="Neutral",
            to_grade="Buy",
            provider="yfinance",
        )
    ]
    catalysts = extract_catalysts([], [], [], actions)
    assert any("Analyst upgrade" in catalyst for catalyst in catalysts)


def test_geopolitical_and_regulation_catalysts():
    news = [
        NewsItem(
            title="Export restrictions and tariff concerns in China market",
            summary="Regulatory compliance review underway.",
            source="Test",
            provider="rss",
        )
    ]
    catalysts = extract_catalysts(news, [], [], [])
    assert any("Geopolitical" in catalyst for catalyst in catalysts)
    assert any("Regulatory" in catalyst for catalyst in catalysts)


def test_litigation_risk_factor():
    news = [
        NewsItem(
            title="Company faces class action lawsuit",
            summary="Litigation risk increases.",
            source="Test",
            provider="yfinance",
        )
    ]
    risks = extract_risk_factors(news, [], [])
    assert "litigation" in risks
