from tradingai.predict.adapters.ml_technical import adapt_ml_technical_signal


def test_adapt_ml_technical_signal_maps_probability_to_direction():
    payload = {
        "ticker": "AAPL",
        "model": "logistic",
        "horizon_days": 5,
        "probability_up": 0.7,
        "directional_score": 0.4,
        "feature_snapshot": {"rsi": 55.0},
    }
    signal = adapt_ml_technical_signal("AAPL", payload)
    assert signal.agent == "technical"
    assert signal.directional_score == 0.4
    assert signal.bullish_score == 0.7
    assert signal.bearish_score == 0.3
    assert "ML logistic model" in signal.reasons[0]


def test_fetch_technical_signal_with_backend_falls_back(monkeypatch):
    from tradingai.predict.adapters import ml_technical
    from tradingai.predict.adapters.technical import adapt_technical_signal

    monkeypatch.setattr(ml_technical, "get_predict_backend", lambda: "ml")

    def _raise(_ticker):
        raise FileNotFoundError("missing artifact")

    fallback_payload = {
        "symbol": "AAPL",
        "bias": "bullish",
        "signals": [
            {"indicator": "sma_50", "signal": "bullish", "detail": "Close is above SMA 50."},
        ],
    }

    monkeypatch.setattr(ml_technical, "fetch_ml_technical_signal", _raise)
    monkeypatch.setattr(
        ml_technical,
        "fetch_technical_signal",
        lambda ticker: adapt_technical_signal(ticker, fallback_payload),
    )

    signal = ml_technical.fetch_technical_signal_with_backend("AAPL")
    assert signal.directional_score > 0
