from __future__ import annotations

from tradingai.analysis.indicators import (
    DEFAULT_BENCHMARK,
    SUPPORTED_INDICATORS,
    compute_indicators,
    latest_indicator_values,
)
from tradingai.analysis.signals import analyze_technical_setup
from tradingai.providers.registry import get_provider


def _ohlcv_records(df) -> list[dict]:
    records = []
    for timestamp, row in df.iterrows():
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return records


def _fetch_symbol_data(
    symbol: str,
    period: str,
    interval: str,
    *,
    benchmark: str | None = DEFAULT_BENCHMARK,
):
    provider = get_provider()
    df = provider.get_ohlcv(symbol.upper(), period, interval)
    benchmark_df = None
    benchmark_symbol = benchmark.upper() if benchmark else DEFAULT_BENCHMARK
    if benchmark:
        benchmark_df = provider.get_ohlcv(benchmark_symbol, period, interval)
    return df, benchmark_df, benchmark_symbol


def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> dict:
    """Fetch normalized OHLCV history for a symbol."""
    provider = get_provider()
    df = provider.get_ohlcv(symbol.upper(), period, interval)
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "rows": len(df),
        "data": _ohlcv_records(df),
    }


def calculate_indicators(
    symbol: str,
    indicators: list[str],
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict:
    """Compute requested technical indicators for a symbol."""
    normalized = [indicator.lower().strip() for indicator in indicators]
    df, benchmark_df, benchmark_symbol = _fetch_symbol_data(
        symbol,
        period,
        interval,
        benchmark=benchmark if "relative_strength" in normalized else None,
    )
    values = latest_indicator_values(
        df,
        indicators,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark_symbol,
    )
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "benchmark": benchmark_symbol if "relative_strength" in normalized else None,
        "indicators": values,
        "supported_indicators": sorted(SUPPORTED_INDICATORS),
    }


def analyze_technical_setup_tool(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict:
    """Return a technical analysis summary with explicit signal labels."""
    df, benchmark_df, benchmark_symbol = _fetch_symbol_data(
        symbol,
        period,
        interval,
        benchmark=benchmark,
    )
    analysis = analyze_technical_setup(
        df,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark_symbol,
    )
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "benchmark": benchmark_symbol,
        **analysis,
    }


def compare_technical_signals(
    symbols: list[str],
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict:
    """Compare technical signals across multiple symbols."""
    if not symbols:
        raise ValueError("At least one symbol must be provided.")

    provider = get_provider()
    benchmark_df = provider.get_ohlcv(benchmark.upper(), period, interval)
    benchmark_symbol = benchmark.upper()

    comparisons = []
    for symbol in symbols:
        df = provider.get_ohlcv(symbol.upper(), period, interval)
        analysis = analyze_technical_setup(
            df,
            benchmark_df=benchmark_df,
            benchmark_symbol=benchmark_symbol,
        )
        comparisons.append(
            {
                "symbol": symbol.upper(),
                "bias": analysis["bias"],
                "close": analysis["close"],
                "summary": analysis["summary"],
                "signals": analysis["signals"],
            }
        )

    return {
        "period": period,
        "interval": interval,
        "benchmark": benchmark_symbol,
        "comparisons": comparisons,
    }
