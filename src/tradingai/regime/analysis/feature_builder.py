from __future__ import annotations

from tradingai.regime.models import MacroFeatures


def build_feature_signals(features: MacroFeatures) -> list[dict]:
    signals: list[dict] = []

    if features.vix is not None:
        high_vol = features.vix >= 25 or (
            features.vix_percentile_1y is not None and features.vix_percentile_1y >= 75
        )
        signals.append(
            {
                "component": "volatility",
                "signal": "high_volatility" if high_vol else "normal_volatility",
                "detail": f"VIX is {features.vix:.2f}"
                + (
                    f" ({features.vix_percentile_1y:.0f}th percentile over 1Y)."
                    if features.vix_percentile_1y is not None
                    else "."
                ),
            }
        )

    if features.spy_close is not None:
        trend = "sideways"
        detail = f"SPY close is {features.spy_close:.2f}."
        if (
            features.spy_sma_200 is not None
            and features.spy_return_6m_pct is not None
            and features.spy_close > features.spy_sma_200
            and features.spy_return_6m_pct > 5
        ):
            trend = "bull"
            detail = (
                f"SPY is above SMA 200 ({features.spy_sma_200:.2f}) with "
                f"{features.spy_return_6m_pct:.2f}% 6-month return."
            )
        elif (
            features.spy_sma_200 is not None
            and features.spy_return_6m_pct is not None
            and features.spy_close < features.spy_sma_200
            and features.spy_return_6m_pct < -5
        ):
            trend = "bear"
            detail = (
                f"SPY is below SMA 200 ({features.spy_sma_200:.2f}) with "
                f"{features.spy_return_6m_pct:.2f}% 6-month return."
            )
        signals.append(
            {
                "component": "trend",
                "signal": trend,
                "detail": detail,
            }
        )

    if features.yield_spread_2s10s_bps is not None:
        signals.append(
            {
                "component": "yield_curve",
                "signal": "inverted" if features.yield_curve_inverted else "normal",
                "detail": (
                    f"2s10s spread is {features.yield_spread_2s10s_bps:.1f} bps."
                ),
            }
        )
    elif features.treasury_10y is not None:
        signals.append(
            {
                "component": "interest_rates",
                "signal": "rate_context",
                "detail": f"10Y treasury proxy is {features.treasury_10y:.2f}%.",
            }
        )

    if features.inflation_breakeven_10y is not None:
        signals.append(
            {
                "component": "inflation",
                "signal": "inflation_expectations",
                "detail": (
                    f"10Y breakeven inflation is {features.inflation_breakeven_10y:.2f}%."
                ),
            }
        )
    if features.cpi_yoy_pct is not None:
        signals.append(
            {
                "component": "inflation",
                "signal": "cpi_yoy",
                "detail": f"CPI year-over-year change is {features.cpi_yoy_pct:.2f}%.",
            }
        )

    if features.credit_spread_hy_oas is not None:
        stressed = (
            features.credit_spread_percentile_1y is not None
            and features.credit_spread_percentile_1y >= 75
        )
        signals.append(
            {
                "component": "credit_spreads",
                "signal": "stressed" if stressed else "normal",
                "detail": (
                    f"HY OAS is {features.credit_spread_hy_oas:.2f}%"
                    + (
                        f" ({features.credit_spread_percentile_1y:.0f}th percentile)."
                        if features.credit_spread_percentile_1y is not None
                        else "."
                    )
                ),
            }
        )
    elif features.hyg_lqd_ratio is not None:
        signals.append(
            {
                "component": "credit_spreads",
                "signal": "proxy_ratio",
                "detail": f"HYG/LQD ratio is {features.hyg_lqd_ratio:.4f}.",
            }
        )

    return signals


def features_to_dict(features: MacroFeatures) -> dict:
    payload = features.model_dump()
    payload["sources_used"] = features.sources_used
    return payload
