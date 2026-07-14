from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

SUPPORTED_INDICATORS = {
    "rsi",
    "macd",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "bbands",
    "atr",
    "momentum",
    "volatility",
    "volume_anomaly",
    "relative_strength",
}

DEFAULT_BENCHMARK = "SPY"
MOMENTUM_LENGTH = 10
VOLATILITY_WINDOW = 20
VOLUME_LOOKBACK = 20
RS_LOOKBACK = 20
TRADING_DAYS_PER_YEAR = 252

MACD_LINE = "MACD_12_26_9"
MACD_SIGNAL = "MACDs_12_26_9"
MACD_HISTOGRAM = "MACDh_12_26_9"
BB_LOWER = "BBL_20_2.0_2.0"
BB_MIDDLE = "BBM_20_2.0_2.0"
BB_UPPER = "BBU_20_2.0_2.0"
MOMENTUM_COLUMN = f"ROC_{MOMENTUM_LENGTH}"


def _validate_indicators(indicators: list[str]) -> list[str]:
    if not indicators:
        raise ValueError("At least one indicator must be requested.")

    normalized = [indicator.lower().strip() for indicator in indicators]
    unknown = sorted(set(normalized) - SUPPORTED_INDICATORS)
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_INDICATORS))
        raise ValueError(
            f"Unsupported indicators: {', '.join(unknown)}. Supported: {supported}."
        )
    return normalized


def _align_benchmark(
    symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame
) -> pd.DataFrame:
    aligned = benchmark_df.reindex(symbol_df.index, method="ffill")
    if aligned["close"].isna().all():
        raise ValueError("Unable to align benchmark data with the symbol history.")
    return aligned


def compute_indicators(
    df: pd.DataFrame,
    indicators: list[str],
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
) -> pd.DataFrame:
    requested = _validate_indicators(indicators)
    result = df.copy()

    if "relative_strength" in requested and benchmark_df is None:
        raise ValueError(
            "relative_strength requires benchmark data. "
            f"Provide benchmark_df or pass benchmark='{DEFAULT_BENCHMARK}' to the tool."
        )

    for indicator in requested:
        if indicator == "rsi":
            result["rsi"] = ta.rsi(result["close"], length=14)
        elif indicator == "macd":
            macd = ta.macd(result["close"], fast=12, slow=26, signal=9)
            if macd is not None:
                result = result.join(macd)
        elif indicator == "sma_20":
            result["sma_20"] = ta.sma(result["close"], length=20)
        elif indicator == "sma_50":
            result["sma_50"] = ta.sma(result["close"], length=50)
        elif indicator == "sma_200":
            result["sma_200"] = ta.sma(result["close"], length=200)
        elif indicator == "ema_12":
            result["ema_12"] = ta.ema(result["close"], length=12)
        elif indicator == "ema_26":
            result["ema_26"] = ta.ema(result["close"], length=26)
        elif indicator == "bbands":
            bands = ta.bbands(result["close"], length=20, std=2)
            if bands is not None:
                result = result.join(bands)
        elif indicator == "atr":
            result["atr"] = ta.atr(
                result["high"], result["low"], result["close"], length=14
            )
        elif indicator == "momentum":
            roc = ta.roc(result["close"], length=MOMENTUM_LENGTH)
            if roc is not None:
                result[MOMENTUM_COLUMN] = roc
        elif indicator == "volatility":
            returns = result["close"].pct_change()
            result["volatility"] = (
                returns.rolling(VOLATILITY_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
            )
        elif indicator == "volume_anomaly":
            volume_sma = ta.sma(result["volume"], length=VOLUME_LOOKBACK)
            volume_mean = result["volume"].rolling(VOLUME_LOOKBACK).mean()
            volume_std = result["volume"].rolling(VOLUME_LOOKBACK).std()
            result["volume_sma"] = volume_sma
            result["volume_ratio"] = result["volume"] / volume_sma
            result["volume_zscore"] = (result["volume"] - volume_mean) / volume_std
        elif indicator == "relative_strength":
            aligned_benchmark = _align_benchmark(result, benchmark_df)
            symbol_return = result["close"].pct_change(RS_LOOKBACK)
            benchmark_return = aligned_benchmark["close"].pct_change(RS_LOOKBACK)
            result["rs_spread"] = (symbol_return - benchmark_return) * 100
            symbol_base = result["close"] / result["close"].iloc[0]
            benchmark_base = (
                aligned_benchmark["close"] / aligned_benchmark["close"].iloc[0]
            )
            result["rs_line"] = (symbol_base / benchmark_base) * 100
            result["benchmark_symbol"] = benchmark_symbol

    return result


def latest_indicator_values(
    df: pd.DataFrame,
    indicators: list[str],
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
) -> dict:
    requested = _validate_indicators(indicators)
    enriched = compute_indicators(
        df,
        requested,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark_symbol,
    )
    latest = enriched.iloc[-1]

    values: dict[str, float | dict[str, float | str | None] | None] = {}
    for indicator in requested:
        if indicator == "macd":
            values["macd"] = {
                "macd": _safe_float(latest.get(MACD_LINE)),
                "signal": _safe_float(latest.get(MACD_SIGNAL)),
                "histogram": _safe_float(latest.get(MACD_HISTOGRAM)),
            }
        elif indicator == "bbands":
            values["bbands"] = {
                "lower": _safe_float(latest.get(BB_LOWER)),
                "middle": _safe_float(latest.get(BB_MIDDLE)),
                "upper": _safe_float(latest.get(BB_UPPER)),
            }
        elif indicator == "momentum":
            values["momentum"] = {
                "roc": _safe_float(latest.get(MOMENTUM_COLUMN)),
                "length": MOMENTUM_LENGTH,
            }
        elif indicator == "volatility":
            values["volatility"] = {
                "annualized_pct": _safe_float(latest.get("volatility")),
                "window": VOLATILITY_WINDOW,
            }
        elif indicator == "volume_anomaly":
            values["volume_anomaly"] = {
                "volume": _safe_float(latest.get("volume")),
                "average_volume": _safe_float(latest.get("volume_sma")),
                "volume_ratio": _safe_float(latest.get("volume_ratio")),
                "volume_zscore": _safe_float(latest.get("volume_zscore")),
                "lookback": VOLUME_LOOKBACK,
            }
        elif indicator == "relative_strength":
            values["relative_strength"] = {
                "benchmark": benchmark_symbol,
                "rs_spread_pct": _safe_float(latest.get("rs_spread")),
                "rs_line": _safe_float(latest.get("rs_line")),
                "lookback": RS_LOOKBACK,
            }
        else:
            values[indicator] = _safe_float(latest.get(indicator))

    return values


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
