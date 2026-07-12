import numpy as np
import pandas as pd
import pytest

from tradingai.analysis.indicators import (
    SUPPORTED_INDICATORS,
    compute_indicators,
    latest_indicator_values,
)
from tradingai.analysis.signals import analyze_technical_setup


def _sample_ohlcv(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(np.linspace(100, 150, rows), index=index)
    volume = np.full(rows, 1_000_000)
    volume[-1] = 2_500_000
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _sample_benchmark(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = pd.Series(np.linspace(400, 420, rows), index=index)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(rows, 5_000_000),
        },
        index=index,
    )


def test_supported_indicator_set():
    assert "rsi" in SUPPORTED_INDICATORS
    assert "macd" in SUPPORTED_INDICATORS
    assert "momentum" in SUPPORTED_INDICATORS
    assert "volatility" in SUPPORTED_INDICATORS
    assert "volume_anomaly" in SUPPORTED_INDICATORS
    assert "relative_strength" in SUPPORTED_INDICATORS


def test_compute_indicators_adds_requested_columns():
    df = _sample_ohlcv()
    result = compute_indicators(df, ["rsi", "sma_20", "atr"])

    assert "rsi" in result.columns
    assert "sma_20" in result.columns
    assert "atr" in result.columns
    assert result["rsi"].iloc[-1] is not None


def test_latest_indicator_values_returns_numeric_payload():
    df = _sample_ohlcv()
    values = latest_indicator_values(df, ["rsi", "macd", "bbands"])

    assert isinstance(values["rsi"], float)
    assert set(values["macd"].keys()) == {"macd", "signal", "histogram"}
    assert set(values["bbands"].keys()) == {"lower", "middle", "upper"}


def test_new_indicator_payload_shapes():
    df = _sample_ohlcv()
    benchmark_df = _sample_benchmark()
    values = latest_indicator_values(
        df,
        ["momentum", "volatility", "volume_anomaly", "relative_strength"],
        benchmark_df=benchmark_df,
        benchmark_symbol="SPY",
    )

    assert "roc" in values["momentum"]
    assert "annualized_pct" in values["volatility"]
    assert "volume_ratio" in values["volume_anomaly"]
    assert values["relative_strength"]["benchmark"] == "SPY"
    assert values["volume_anomaly"]["volume_ratio"] > 2.0


def test_analyze_technical_setup_returns_summary_and_bias():
    df = _sample_ohlcv()
    benchmark_df = _sample_benchmark()
    analysis = analyze_technical_setup(
        df,
        benchmark_df=benchmark_df,
        benchmark_symbol="SPY",
    )

    assert analysis["bias"] in {"bullish", "bearish", "neutral"}
    assert isinstance(analysis["summary"], str)
    assert len(analysis["signals"]) > 0
    assert "not financial advice" in analysis["summary"].lower()
    indicator_names = {signal["indicator"] for signal in analysis["signals"]}
    assert "momentum" in indicator_names
    assert "volume_anomaly" in indicator_names
    assert "relative_strength" in indicator_names


def test_compute_indicators_rejects_unknown_indicator():
    df = _sample_ohlcv()
    with pytest.raises(ValueError, match="Unsupported indicators"):
        compute_indicators(df, ["made_up_indicator"])


def test_relative_strength_requires_benchmark_data():
    df = _sample_ohlcv()
    with pytest.raises(ValueError, match="relative_strength requires benchmark data"):
        compute_indicators(df, ["relative_strength"])
