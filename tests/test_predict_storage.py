from datetime import datetime, timezone
from unittest.mock import patch

from tradingai.predict.fusion.scorer import fuse_agent_signals
from tradingai.predict.models import AgentSignal
from tradingai.predict.storage.repository import save_prediction


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
            raw={"sentiment": 0.6, "catalysts": ["beat estimates"], "risk_factors": []},
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


def test_save_prediction_uses_postgres_repository():
    report = _report()
    with (
        patch("tradingai.predict.storage.repository.ensure_stock", return_value={"sector": "Technology"}),
        patch("tradingai.predict.storage.repository.predictions_repo.save_prediction_rows", return_value=[1, 2, 3]) as mock_save,
    ):
        ids = save_prediction(report)
    assert ids == [1, 2, 3]
    mock_save.assert_called_once()
    rows = mock_save.call_args.args[0]
    assert len(rows) == 3
    assert rows[0]["sector"] == "Technology"
    assert rows[0]["prediction_type"] in {
        "earnings_driven_rally",
        "general_directional",
        "product_catalyst_rally",
        "analyst_upgrade_rally",
    }
