from __future__ import annotations

import pandas as pd

from tradingai.regime.analysis.classifier import _trend_label, _vol_label, classify_regime
from tradingai.regime.models import MacroFeatures, MarketFactors


GROWTH_VALUE_THRESHOLD_PP = 3.0


def _six_month_return_pct(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    lookback = min(126, len(clean) - 1)
    if lookback <= 0:
        return None
    return float((clean.iloc[-1] / clean.iloc[-lookback] - 1.0) * 100.0)


def _hyg_lqd_bottom_quartile(hyg: pd.Series, lqd: pd.Series) -> bool:
    """True when the latest HYG/LQD ratio is in the bottom quartile of the last year."""
    if hyg.empty or lqd.empty:
        return False
    aligned = pd.concat([hyg.rename("hyg"), lqd.rename("lqd")], axis=1).dropna()
    if aligned.empty:
        return False
    aligned = aligned.tail(252)
    ratio = aligned["hyg"] / aligned["lqd"].replace(0, pd.NA)
    ratio = ratio.dropna()
    if len(ratio) < 20:
        return False
    latest = float(ratio.iloc[-1])
    q25 = float(ratio.quantile(0.25))
    return latest <= q25


def classify_growth_value(
    *,
    xlk_return_6m_pct: float | None,
    xlu_return_6m_pct: float | None,
) -> str:
    if xlk_return_6m_pct is None or xlu_return_6m_pct is None:
        return "neutral"
    spread = xlk_return_6m_pct - xlu_return_6m_pct
    if spread > GROWTH_VALUE_THRESHOLD_PP:
        return "growth"
    if spread < -GROWTH_VALUE_THRESHOLD_PP:
        return "value"
    return "neutral"


def classify_market_factors(
    features: MacroFeatures,
    *,
    xlk_return_6m_pct: float | None = None,
    xlu_return_6m_pct: float | None = None,
    hyg_series: pd.Series | None = None,
    lqd_series: pd.Series | None = None,
) -> MarketFactors:
    equity_trend = _trend_label(features)
    vol_raw = _vol_label(features)
    vol = "high" if vol_raw == "high_volatility" else "normal"
    market_regime = classify_regime(features)

    if equity_trend == "bull" and vol == "normal":
        risk = "risk_on"
    elif equity_trend == "bear" or market_regime == "high_volatility_bearish":
        risk = "risk_off"
    else:
        risk = "neutral"

    rates = "tightening" if features.yield_curve_inverted else "neutral"

    credit = "normal"
    if features.credit_spread_percentile_1y is not None and features.credit_spread_percentile_1y >= 75:
        credit = "stressed"
    elif features.credit_spread_percentile_1y is None and hyg_series is not None and lqd_series is not None:
        if _hyg_lqd_bottom_quartile(hyg_series, lqd_series):
            credit = "stressed"

    growth_value = classify_growth_value(
        xlk_return_6m_pct=xlk_return_6m_pct,
        xlu_return_6m_pct=xlu_return_6m_pct,
    )

    return MarketFactors(
        equity_trend=equity_trend,
        vol=vol,
        risk=risk,
        rates=rates,
        credit=credit,
        growth_value=growth_value,
        ai_cycle=None,
        geopolitics=None,
    )


def six_month_return_from_series(series: pd.Series) -> float | None:
    return _six_month_return_pct(series)
