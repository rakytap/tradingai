from __future__ import annotations

import numpy as np
import pandas as pd

from tradingai.predict.adapters.regime import STANCE_SCORES, fetch_regime_signal
from tradingai.regime.analysis.classifier import classify_regime
from tradingai.regime.analysis.historical import macro_features_as_of
from tradingai.regime.analysis.market_factors import (
    classify_growth_value,
    classify_market_factors,
)
from tradingai.regime.analysis.sector_board import classify_sector_board, regime_to_stance
from tradingai.regime.models import MarketFactors, RegimeState, SectorStance
from tradingai.regime.sector_proxies import (
    DEFAULT_REGIME_PROXY,
    sector_regime_proxy,
)


def _series(start: str, values: list[float]) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index)


def _bull_features():
    trend = _series("2023-01-01", np.linspace(100, 160, 300).tolist())
    vix_values = np.linspace(12.0, 22.0, 300).tolist()
    vix_values[-1] = 15.0
    return macro_features_as_of(
        spy=trend,
        vix=_series("2023-01-01", vix_values),
        hyg=_series("2023-01-01", np.linspace(75, 78, 300).tolist()),
        lqd=_series("2023-01-01", np.linspace(110, 112, 300).tolist()),
        as_of=trend.index[-1].date(),
        sources_used=["test"],
    )


def _bear_high_vol_features():
    trend = _series("2023-01-01", np.linspace(160, 100, 300).tolist())
    vix_values = np.linspace(20.0, 35.0, 300).tolist()
    vix_values[-1] = 30.0
    return macro_features_as_of(
        spy=trend,
        vix=_series("2023-01-01", vix_values),
        hyg=_series("2023-01-01", np.linspace(75, 70, 300).tolist()),
        lqd=_series("2023-01-01", np.linspace(110, 112, 300).tolist()),
        as_of=trend.index[-1].date(),
        sources_used=["test"],
    )


def test_sector_regime_proxy_map():
    assert sector_regime_proxy("Technology") == "XLK"
    assert sector_regime_proxy("ENERGY") == "XLE"
    assert sector_regime_proxy("ETF") == DEFAULT_REGIME_PROXY
    assert sector_regime_proxy(None) == DEFAULT_REGIME_PROXY


def test_classify_market_factors_risk_on():
    features = _bull_features()
    factors = classify_market_factors(features, xlk_return_6m_pct=10.0, xlu_return_6m_pct=2.0)
    assert factors.equity_trend == "bull"
    assert factors.vol == "normal"
    assert factors.risk == "risk_on"
    assert factors.rates == "neutral"
    assert factors.growth_value == "growth"
    assert factors.ai_cycle is None
    assert factors.geopolitics is None


def test_classify_market_factors_risk_off_high_vol_bear():
    features = _bear_high_vol_features()
    assert classify_regime(features) == "high_volatility_bearish"
    factors = classify_market_factors(features)
    assert factors.risk == "risk_off"
    assert factors.vol == "high"
    assert factors.equity_trend == "bear"


def test_classify_market_factors_tightening_and_credit_stressed():
    features = _bull_features()
    features.yield_curve_inverted = True
    features.credit_spread_percentile_1y = 80.0
    factors = classify_market_factors(features)
    assert factors.rates == "tightening"
    assert factors.credit == "stressed"


def test_classify_growth_value_thresholds():
    assert classify_growth_value(xlk_return_6m_pct=8.0, xlu_return_6m_pct=2.0) == "growth"
    assert classify_growth_value(xlk_return_6m_pct=1.0, xlu_return_6m_pct=6.0) == "value"
    assert classify_growth_value(xlk_return_6m_pct=5.0, xlu_return_6m_pct=4.0) == "neutral"


def test_hyg_lqd_bottom_quartile_marks_credit_stressed():
    features = _bull_features()
    features.credit_spread_percentile_1y = None
    # Falling HYG vs flat LQD → low ratio at end.
    hyg = _series("2023-01-01", np.linspace(90, 60, 300).tolist())
    lqd = _series("2023-01-01", np.full(300, 100.0).tolist())
    factors = classify_market_factors(features, hyg_series=hyg, lqd_series=lqd)
    assert factors.credit == "stressed"


def test_regime_to_stance_mapping():
    assert regime_to_stance("bull") == "bullish"
    assert regime_to_stance("bull_high_volatility") == "bullish"
    assert regime_to_stance("bear") == "bearish"
    assert regime_to_stance("high_volatility_bearish") == "bearish"
    assert regime_to_stance("sideways") == "neutral"
    assert regime_to_stance("high_volatility") == "neutral"


def test_classify_sector_board_technology_bullish():
    board = classify_sector_board({"XLK": _bull_features()})
    assert len(board) == 1
    assert board[0].sector == "Technology"
    assert board[0].proxy == "XLK"
    assert board[0].stance == "bullish"


def test_regime_state_public_dict_shape():
    state = RegimeState(
        market=MarketFactors(
            equity_trend="bull",
            vol="normal",
            risk="risk_on",
            rates="neutral",
            credit="normal",
            growth_value="growth",
        ),
        sectors=[
            SectorStance(
                sector="Technology",
                proxy="XLK",
                regime="bull",
                stance="bullish",
            )
        ],
        market_regime="bull",
        confidence=0.7,
        evidence={"sector_proxies": {"Technology": "XLK"}},
    )
    payload = state.to_public_dict()
    assert set(payload["market"]) >= {
        "equity_trend",
        "vol",
        "risk",
        "rates",
        "credit",
        "growth_value",
        "ai_cycle",
        "geopolitics",
    }
    assert payload["sectors"][0]["stance"] == "bullish"
    assert payload["market_regime"] == "bull"
    assert state.stance_for_sector("Technology") == "bullish"


def test_fetch_regime_signal_uses_sector_stance(monkeypatch):
    state = RegimeState(
        market=MarketFactors(
            equity_trend="bull",
            vol="normal",
            risk="risk_on",
            rates="neutral",
            credit="normal",
            growth_value="growth",
        ),
        sectors=[
            SectorStance(
                sector="Technology",
                proxy="XLK",
                regime="bull",
                stance="bullish",
            )
        ],
        market_regime="bull",
        confidence=0.66,
        evidence={},
    )
    monkeypatch.setattr(
        "tradingai.regime.tools.regime.load_or_build_regime_state",
        lambda: state,
    )
    monkeypatch.setattr(
        "tradingai.predict.adapters.regime.resolve_regime_proxy_for_ticker",
        lambda ticker, database_url=None: ("Technology", "XLK"),
    )
    signal = fetch_regime_signal("AAPL")
    assert signal.directional_score == STANCE_SCORES["bullish"]
    assert signal.raw["stance"] == "bullish"
    assert signal.raw["regime"] == "bullish"
    assert "Technology" in signal.reasons[0]
