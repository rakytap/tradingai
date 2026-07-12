from tradingai.predict.fusion.scorer import fuse_agent_signals
from tradingai.predict.models import AgentSignal


def _signals() -> list[AgentSignal]:
    return [
        AgentSignal(
            agent="technical",
            ticker="AAPL",
            directional_score=0.42,
            bullish_score=0.71,
            bearish_score=0.29,
            confidence=0.57,
            reasons=["Price above 50-day MA"],
            raw={},
        ),
        AgentSignal(
            agent="news",
            ticker="AAPL",
            directional_score=0.30,
            bullish_score=0.65,
            bearish_score=0.35,
            confidence=0.81,
            reasons=["Positive earnings revisions"],
            raw={},
        ),
        AgentSignal(
            agent="regime",
            ticker=None,
            directional_score=0.18,
            bullish_score=0.59,
            bearish_score=0.41,
            confidence=0.74,
            reasons=["Market regime bullish"],
            raw={},
        ),
    ]


def test_fuse_agent_signals_combined_score():
    report = fuse_agent_signals("AAPL", _signals())
    assert report.ticker == "AAPL"
    assert report.fusion["technical_score"] == 0.42
    assert report.fusion["news_score"] == 0.30
    assert report.fusion["regime_score"] == 0.18
    assert report.fusion["combined_score"] > 0
    assert 0.3 <= report.confidence <= 0.95


def test_fuse_agent_signals_multi_horizon():
    report = fuse_agent_signals("AAPL", _signals())
    horizons = {item.horizon_days: item for item in report.horizons}
    assert set(horizons) == {1, 5, 20}
    assert 0.0 <= horizons[5].probability_up <= 1.0
    assert horizons[5].expected_return_pct != 0.0


def test_fuse_agent_signals_reasons():
    report = fuse_agent_signals("AAPL", _signals())
    assert "Price above 50-day MA" in report.reasons
    assert "Positive earnings revisions" in report.reasons
