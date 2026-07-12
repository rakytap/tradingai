from datetime import datetime, timezone

from tradingai.predict.fusion.scorer import fuse_agent_signals
from tradingai.predict.models import AgentSignal
from tradingai.predict.storage.repository import (
    get_prediction_history,
    save_prediction,
)


def _report():
    signals = [
        AgentSignal(
            agent="technical",
            ticker="AAPL",
            directional_score=0.5,
            bullish_score=0.75,
            bearish_score=0.25,
            confidence=0.6,
            reasons=["Bullish technical setup"],
            raw={"bias": "bullish"},
        ),
        AgentSignal(
            agent="news",
            ticker="AAPL",
            directional_score=0.2,
            bullish_score=0.6,
            bearish_score=0.4,
            confidence=0.7,
            reasons=["Positive catalyst"],
            raw={"sentiment": 0.6},
        ),
        AgentSignal(
            agent="regime",
            ticker=None,
            directional_score=0.8,
            bullish_score=0.9,
            bearish_score=0.1,
            confidence=0.75,
            reasons=["Bull regime"],
            raw={"regime": "bull"},
        ),
    ]
    report = fuse_agent_signals("AAPL", signals)
    report.as_of = datetime(2026, 1, 10, tzinfo=timezone.utc)
    return report


def test_save_and_query_prediction(tmp_path):
    db_path = str(tmp_path / "predictions.sqlite3")
    report = _report()
    prediction_id = save_prediction(report, db_path=db_path)
    assert prediction_id > 0

    history = get_prediction_history("AAPL", limit=5, db_path=db_path)
    assert len(history) == 1
    assert history[0]["ticker"] == "AAPL"
    assert history[0]["combined_score"] == report.fusion["combined_score"]
