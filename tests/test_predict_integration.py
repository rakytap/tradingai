from unittest.mock import patch

import pytest

from tradingai.predict.adapters.news import adapt_news_signal
from tradingai.predict.adapters.regime import adapt_regime_signal
from tradingai.predict.adapters.technical import adapt_technical_signal
from tradingai.predict.tools.predict import generate_prediction_tool


def test_generate_prediction_end_to_end_mocked():

    technical_payload = {
        "symbol": "AAPL",
        "signals": [
            {"indicator": "sma_50", "signal": "bullish", "detail": "Price above 50-day MA"},
        ],
    }
    news_payload = {
        "ticker": "AAPL",
        "sentiment": 0.65,
        "confidence": 0.81,
        "catalysts": ["Positive earnings revisions"],
        "risk_factors": [],
    }
    regime_payload = {
        "regime": "bull",
        "confidence": 0.74,
        "signals": [{"component": "trend", "signal": "bull", "detail": "Market regime bullish"}],
    }

    with (
        patch(
            "tradingai.predict.tools.predict.fetch_technical_signal",
            return_value=adapt_technical_signal("AAPL", technical_payload),
        ),
        patch(
            "tradingai.predict.tools.predict.fetch_news_signal",
            return_value=adapt_news_signal("AAPL", news_payload),
        ),
        patch(
            "tradingai.predict.tools.predict.fetch_regime_signal",
            return_value=adapt_regime_signal(regime_payload),
        ),
        patch(
            "tradingai.predict.tools.predict.save_prediction",
            return_value=[1, 2, 3],
        ),
    ):
        payload = generate_prediction_tool("AAPL")

    assert payload["ticker"] == "AAPL"
    assert payload["prediction_ids"] == [1, 2, 3]
    assert payload["fusion"]["combined_score"] > 0
    assert len(payload["horizons"]) == 3
    assert "Positive earnings revisions" in payload["reasons"]


@pytest.mark.integration
def test_generate_prediction_live_round_trip():
    try:
        payload = generate_prediction_tool("AAPL")
    except Exception as exc:
        pytest.skip(f"Live prediction unavailable: {exc}")
    assert payload["ticker"] == "AAPL"
    assert len(payload["prediction_ids"]) == 3

