from __future__ import annotations

import pandas as pd

from tradingai.analysis.indicators import (
    BB_LOWER,
    BB_UPPER,
    MACD_HISTOGRAM,
    MACD_LINE,
    MACD_SIGNAL,
    MOMENTUM_COLUMN,
    SUPPORTED_INDICATORS,
    VOLUME_LOOKBACK,
    compute_indicators,
)


def analyze_technical_setup(
    df: pd.DataFrame,
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
) -> dict:
    indicators = sorted(SUPPORTED_INDICATORS)
    enriched = compute_indicators(
        df,
        indicators,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark_symbol,
    )
    latest = enriched.iloc[-1]
    previous = enriched.iloc[-2] if len(enriched) > 1 else latest

    close = float(latest["close"])
    signals: list[dict[str, str]] = []

    rsi = latest.get("rsi")
    if rsi is not None and not pd.isna(rsi):
        rsi_value = float(rsi)
        if rsi_value >= 70:
            signals.append(
                {
                    "indicator": "rsi",
                    "signal": "overbought",
                    "detail": f"RSI is {rsi_value:.2f}, above the 70 overbought threshold.",
                }
            )
        elif rsi_value <= 30:
            signals.append(
                {
                    "indicator": "rsi",
                    "signal": "oversold",
                    "detail": f"RSI is {rsi_value:.2f}, below the 30 oversold threshold.",
                }
            )
        else:
            signals.append(
                {
                    "indicator": "rsi",
                    "signal": "neutral",
                    "detail": f"RSI is {rsi_value:.2f}, within the neutral range.",
                }
            )

    for sma_name in ("sma_20", "sma_50", "sma_200"):
        sma_value = latest.get(sma_name)
        if sma_value is None or pd.isna(sma_value):
            continue
        sma_value = float(sma_value)
        if close > sma_value:
            signals.append(
                {
                    "indicator": sma_name,
                    "signal": "bullish",
                    "detail": f"Close {close:.2f} is above {sma_name.upper()} {sma_value:.2f}.",
                }
            )
        else:
            signals.append(
                {
                    "indicator": sma_name,
                    "signal": "bearish",
                    "detail": f"Close {close:.2f} is below {sma_name.upper()} {sma_value:.2f}.",
                }
            )

    macd = latest.get(MACD_LINE)
    macd_signal = latest.get(MACD_SIGNAL)
    macd_hist = latest.get(MACD_HISTOGRAM)
    prev_macd = previous.get(MACD_LINE)
    prev_macd_signal = previous.get(MACD_SIGNAL)
    if (
        macd is not None
        and macd_signal is not None
        and not pd.isna(macd)
        and not pd.isna(macd_signal)
    ):
        macd_value = float(macd)
        signal_value = float(macd_signal)
        if (
            prev_macd is not None
            and prev_macd_signal is not None
            and not pd.isna(prev_macd)
            and not pd.isna(prev_macd_signal)
        ):
            if float(prev_macd) <= float(prev_macd_signal) and macd_value > signal_value:
                signals.append(
                    {
                        "indicator": "macd",
                        "signal": "bullish_crossover",
                        "detail": "MACD crossed above its signal line.",
                    }
                )
            elif float(prev_macd) >= float(prev_macd_signal) and macd_value < signal_value:
                signals.append(
                    {
                        "indicator": "macd",
                        "signal": "bearish_crossover",
                        "detail": "MACD crossed below its signal line.",
                    }
                )
            elif macd_value > signal_value:
                signals.append(
                    {
                        "indicator": "macd",
                        "signal": "bullish",
                        "detail": "MACD remains above its signal line.",
                    }
                )
            else:
                signals.append(
                    {
                        "indicator": "macd",
                        "signal": "bearish",
                        "detail": "MACD remains below its signal line.",
                    }
                )

        if macd_hist is not None and not pd.isna(macd_hist):
            hist_value = float(macd_hist)
            direction = "rising" if hist_value > 0 else "falling"
            signals.append(
                {
                    "indicator": "macd_histogram",
                    "signal": direction,
                    "detail": f"MACD histogram is {hist_value:.4f} and {direction}.",
                }
            )

    lower_band = latest.get(BB_LOWER)
    upper_band = latest.get(BB_UPPER)
    if (
        lower_band is not None
        and upper_band is not None
        and not pd.isna(lower_band)
        and not pd.isna(upper_band)
    ):
        band_width = float(upper_band) - float(lower_band)
        if close <= float(lower_band):
            signals.append(
                {
                    "indicator": "bbands",
                    "signal": "near_lower_band",
                    "detail": "Price is at or below the lower Bollinger Band.",
                }
            )
        elif close >= float(upper_band):
            signals.append(
                {
                    "indicator": "bbands",
                    "signal": "near_upper_band",
                    "detail": "Price is at or above the upper Bollinger Band.",
                }
            )
        else:
            signals.append(
                {
                    "indicator": "bbands",
                    "signal": "mid_range",
                    "detail": f"Price is within Bollinger Bands (width {band_width:.2f}).",
                }
            )

    atr = latest.get("atr")
    if atr is not None and not pd.isna(atr):
        atr_value = float(atr)
        signals.append(
            {
                "indicator": "atr",
                "signal": "volatility_context",
                "detail": f"ATR(14) is {atr_value:.2f}, indicating recent volatility.",
            }
        )

    momentum = latest.get(MOMENTUM_COLUMN)
    if momentum is not None and not pd.isna(momentum):
        momentum_value = float(momentum)
        if momentum_value >= 5:
            signals.append(
                {
                    "indicator": "momentum",
                    "signal": "strong_positive",
                    "detail": f"ROC(10) is {momentum_value:.2f}%, showing strong positive momentum.",
                }
            )
        elif momentum_value <= -5:
            signals.append(
                {
                    "indicator": "momentum",
                    "signal": "strong_negative",
                    "detail": f"ROC(10) is {momentum_value:.2f}%, showing strong negative momentum.",
                }
            )
        else:
            signals.append(
                {
                    "indicator": "momentum",
                    "signal": "neutral",
                    "detail": f"ROC(10) is {momentum_value:.2f}%, within a neutral momentum range.",
                }
            )

    volatility = latest.get("volatility")
    if volatility is not None and not pd.isna(volatility):
        volatility_value = float(volatility)
        vol_series = enriched["volatility"].dropna()
        if not vol_series.empty:
            median_vol = float(vol_series.median())
            if volatility_value >= median_vol * 1.25:
                signal = "elevated"
                detail = (
                    f"Annualized volatility is {volatility_value:.2f}%, above the recent median "
                    f"of {median_vol:.2f}%."
                )
            elif volatility_value <= median_vol * 0.75:
                signal = "compressed"
                detail = (
                    f"Annualized volatility is {volatility_value:.2f}%, below the recent median "
                    f"of {median_vol:.2f}%."
                )
            else:
                signal = "normal"
                detail = (
                    f"Annualized volatility is {volatility_value:.2f}%, near the recent median "
                    f"of {median_vol:.2f}%."
                )
            signals.append(
                {
                    "indicator": "volatility",
                    "signal": signal,
                    "detail": detail,
                }
            )

    volume_ratio = latest.get("volume_ratio")
    volume_zscore = latest.get("volume_zscore")
    if volume_ratio is not None and not pd.isna(volume_ratio):
        ratio_value = float(volume_ratio)
        zscore_value = (
            float(volume_zscore) if volume_zscore is not None and not pd.isna(volume_zscore) else None
        )
        if ratio_value >= 2.0 or (zscore_value is not None and zscore_value >= 2.0):
            signals.append(
                {
                    "indicator": "volume_anomaly",
                    "signal": "unusual_high",
                    "detail": (
                        f"Volume is {ratio_value:.2f}x the {VOLUME_LOOKBACK}-day average, "
                        "indicating unusually high participation."
                    ),
                }
            )
        elif ratio_value <= 0.5:
            signals.append(
                {
                    "indicator": "volume_anomaly",
                    "signal": "unusual_low",
                    "detail": (
                        f"Volume is {ratio_value:.2f}x the recent average, indicating unusually low "
                        "participation."
                    ),
                }
            )
        else:
            signals.append(
                {
                    "indicator": "volume_anomaly",
                    "signal": "normal",
                    "detail": f"Volume is {ratio_value:.2f}x the recent average, within a normal range.",
                }
            )

    rs_spread = latest.get("rs_spread")
    if rs_spread is not None and not pd.isna(rs_spread):
        spread_value = float(rs_spread)
        if spread_value > 1:
            signals.append(
                {
                    "indicator": "relative_strength",
                    "signal": "outperforming",
                    "detail": (
                        f"Outperforming {benchmark_symbol} by {spread_value:.2f}% over the lookback "
                        "window."
                    ),
                }
            )
        elif spread_value < -1:
            signals.append(
                {
                    "indicator": "relative_strength",
                    "signal": "underperforming",
                    "detail": (
                        f"Underperforming {benchmark_symbol} by {abs(spread_value):.2f}% over the "
                        "lookback window."
                    ),
                }
            )
        else:
            signals.append(
                {
                    "indicator": "relative_strength",
                    "signal": "inline",
                    "detail": (
                        f"Performance is broadly in line with {benchmark_symbol} "
                        f"({spread_value:.2f}% spread)."
                    ),
                }
            )

    bias = _overall_bias(signals)
    summary = _build_summary(close, bias, signals)

    return {
        "close": close,
        "signals": signals,
        "bias": bias,
        "summary": summary,
    }


def _overall_bias(signals: list[dict[str, str]]) -> str:
    bullish = 0
    bearish = 0
    bullish_terms = {
        "bullish",
        "bullish_crossover",
        "oversold",
        "strong_positive",
        "outperforming",
        "unusual_high",
    }
    bearish_terms = {
        "bearish",
        "bearish_crossover",
        "overbought",
        "strong_negative",
        "underperforming",
    }

    for signal in signals:
        label = signal["signal"]
        if label in bullish_terms:
            bullish += 1
        elif label in bearish_terms:
            bearish += 1

    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def _build_summary(close: float, bias: str, signals: list[dict[str, str]]) -> str:
    highlights = [signal["detail"] for signal in signals[:3]]
    highlight_text = " ".join(highlights)
    disclaimer = (
        "This is technical analysis for informational purposes only and is not financial advice."
    )
    return (
        f"Overall technical bias is {bias} with close at {close:.2f}. "
        f"{highlight_text} {disclaimer}"
    )
