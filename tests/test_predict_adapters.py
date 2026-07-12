from tradingai.predict.adapters.news import adapt_news_signal
from tradingai.predict.adapters.regime import adapt_regime_signal
from tradingai.predict.adapters.technical import adapt_technical_signal


def test_adapt_technical_signal_bullish():
    payload = {
        "symbol": "AAPL",
        "bias": "bullish",
        "signals": [
            {"indicator": "sma_50", "signal": "bullish", "detail": "Close is above SMA 50."},
            {"indicator": "macd", "signal": "bullish_crossover", "detail": "MACD crossed above signal."},
            {"indicator": "rsi", "signal": "neutral", "detail": "RSI is neutral."},
        ],
    }
    signal = adapt_technical_signal("AAPL", payload)
    assert signal.agent == "technical"
    assert signal.directional_score > 0
    assert signal.bullish_score > signal.bearish_score
    assert len(signal.reasons) >= 2


def test_adapt_news_signal_positive_sentiment():
    payload = {
        "ticker": "NVDA",
        "sentiment": 0.78,
        "confidence": 0.81,
        "catalysts": ["Positive earnings revisions", "Analyst upgrade"],
        "risk_factors": ["valuation"],
    }
    signal = adapt_news_signal("NVDA", payload)
    assert signal.directional_score > 0
    assert signal.bullish_score == 0.78
    assert signal.bearish_score == 0.22
    assert signal.confidence == 0.81
    assert "Positive earnings revisions" in signal.reasons


def test_adapt_regime_signal_bull():
    payload = {
        "regime": "bull",
        "confidence": 0.74,
        "signals": [
            {"component": "trend", "signal": "bull", "detail": "Market regime bullish"},
            {"component": "volatility", "signal": "normal_volatility", "detail": "VIX normal."},
        ],
    }
    signal = adapt_regime_signal(payload)
    assert signal.agent == "regime"
    assert signal.ticker is None
    assert signal.directional_score == 0.80
    assert "Market regime bullish" in signal.reasons
