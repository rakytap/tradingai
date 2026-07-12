from tradingai.regime.analysis.feature_builder import build_feature_signals, features_to_dict
from tradingai.regime.analysis.recommendations import STRATEGY_MAP, recommend_strategy
from tradingai.regime.models import MacroFeatures, RegimeReport


def test_build_feature_signals_for_bullish_inputs():
    features = MacroFeatures(
        vix=16.0,
        vix_percentile_1y=35.0,
        spy_close=520.0,
        spy_sma_200=480.0,
        spy_return_6m_pct=10.0,
        treasury_10y=4.2,
        hyg_lqd_ratio=0.72,
        sources_used=["yfinance"],
    )
    signals = build_feature_signals(features)
    trend = next(s for s in signals if s["component"] == "trend")
    vol = next(s for s in signals if s["component"] == "volatility")
    assert trend["signal"] == "bull"
    assert vol["signal"] == "normal_volatility"


def test_build_feature_signals_with_fred_macro_stress():
    features = MacroFeatures(
        vix=22.0,
        vix_percentile_1y=60.0,
        spy_close=430.0,
        spy_sma_200=460.0,
        spy_return_6m_pct=-7.0,
        yield_spread_2s10s_bps=-25.0,
        yield_curve_inverted=True,
        credit_spread_hy_oas=5.5,
        credit_spread_percentile_1y=80.0,
        inflation_breakeven_10y=2.4,
        cpi_yoy_pct=3.1,
        sources_used=["yfinance", "fred"],
    )
    signals = build_feature_signals(features)
    components = {signal["component"]: signal["signal"] for signal in signals}
    assert components["trend"] == "bear"
    assert components["yield_curve"] == "inverted"
    assert components["credit_spreads"] == "stressed"


def test_features_to_dict_includes_sources():
    features = MacroFeatures(vix=20.0, sources_used=["yfinance"])
    payload = features_to_dict(features)
    assert payload["vix"] == 20.0
    assert payload["sources_used"] == ["yfinance"]


def test_strategy_recommendation_mapping():
    for regime, strategy in STRATEGY_MAP.items():
        assert recommend_strategy(regime) == strategy


def test_regime_report_schema_bounds():
    report = RegimeReport(
        regime="bull",
        recommended_strategy="increase_trend_exposure",
        confidence=0.72,
        features={"vix": 18.0},
        signals=[{"component": "trend", "signal": "bull", "detail": "test"}],
    )
    payload = report.to_public_dict()
    assert payload["regime"] == "bull"
    assert payload["recommended_strategy"] == "increase_trend_exposure"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert "as_of" in payload
