import pandas as pd

from tradingai.regime.analysis.classifier import classify_regime, compute_confidence
from tradingai.regime.analysis.feature_builder import build_feature_signals
from tradingai.regime.models import MacroFeatures


def _features(**kwargs) -> MacroFeatures:
    defaults = {
        "vix": 18.0,
        "vix_percentile_1y": 40.0,
        "spy_close": 500.0,
        "spy_sma_50": 490.0,
        "spy_sma_200": 470.0,
        "spy_return_6m_pct": 8.0,
        "sources_used": ["yfinance"],
    }
    defaults.update(kwargs)
    return MacroFeatures(**defaults)


def test_classify_bull():
    assert classify_regime(_features()) == "bull"


def test_classify_bear():
    features = _features(
        spy_close=420.0,
        spy_sma_200=450.0,
        spy_return_6m_pct=-8.0,
    )
    assert classify_regime(features) == "bear"


def test_classify_high_volatility_bearish():
    features = _features(
        vix=30.0,
        vix_percentile_1y=80.0,
        spy_close=420.0,
        spy_sma_200=450.0,
        spy_return_6m_pct=-8.0,
    )
    assert classify_regime(features) == "high_volatility_bearish"


def test_classify_bull_high_volatility():
    features = _features(vix=28.0, vix_percentile_1y=78.0)
    assert classify_regime(features) == "bull_high_volatility"


def test_classify_sideways_high_volatility():
    features = _features(
        vix=27.0,
        vix_percentile_1y=76.0,
        spy_return_6m_pct=1.0,
    )
    assert classify_regime(features) == "high_volatility"


def test_classify_sideways():
    features = _features(spy_return_6m_pct=1.0)
    assert classify_regime(features) == "sideways"


def test_confidence_bounds():
    features = _features()
    signals = build_feature_signals(features)
    confidence = compute_confidence(features, signals)
    assert 0.4 <= confidence <= 0.9


def test_confidence_higher_with_fred_sources():
    features = _features(sources_used=["yfinance", "fred"])
    signals = build_feature_signals(features)
    yfinance_only = compute_confidence(_features(), build_feature_signals(_features()))
    with_fred = compute_confidence(features, signals)
    assert with_fred >= yfinance_only
