from mcp.server.fastmcp import FastMCP

from tradingai.tools import technical

mcp = FastMCP(
    "tradingai",
    instructions=(
        "Stock trading technical analysis tools. "
        "Outputs are informational only and not financial advice."
    ),
)


@mcp.tool()
def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> dict:
    """Fetch normalized OHLCV price history for a stock symbol.

    Args:
        symbol: Ticker symbol, e.g. AAPL.
        period: Lookback period supported by the data provider, e.g. 6mo, 1y.
        interval: Candle interval, e.g. 1d, 1h.
    """
    return technical.get_ohlcv(symbol=symbol, period=period, interval=interval)


@mcp.tool()
def calculate_indicators(
    symbol: str,
    indicators: list[str],
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = "SPY",
) -> dict:
    """Compute technical indicators for a stock symbol.

    Supported indicators:
    rsi, macd, sma_20, sma_50, sma_200, ema_12, ema_26, bbands, atr,
    momentum, volatility, volume_anomaly, relative_strength.

    Use benchmark when requesting relative_strength. Defaults to SPY.

    Args:
        symbol: Ticker symbol, e.g. AAPL.
        indicators: List of indicator names to compute.
        period: Lookback period supported by the data provider.
        interval: Candle interval.
        benchmark: Benchmark ticker for relative_strength, default SPY.
    """
    return technical.calculate_indicators(
        symbol=symbol,
        indicators=indicators,
        period=period,
        interval=interval,
        benchmark=benchmark,
    )


@mcp.tool()
def analyze_technical_setup(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = "SPY",
) -> dict:
    """Analyze trend, momentum, volatility, volume, and relative strength for a symbol.

    Returns structured signals plus a short natural-language summary.
    This is informational technical analysis, not financial advice.

    Args:
        symbol: Ticker symbol, e.g. AAPL.
        period: Lookback period supported by the data provider.
        interval: Candle interval.
        benchmark: Benchmark ticker for relative strength, default SPY.
    """
    return technical.analyze_technical_setup_tool(
        symbol=symbol,
        period=period,
        interval=interval,
        benchmark=benchmark,
    )


@mcp.tool()
def compare_technical_signals(
    symbols: list[str],
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = "SPY",
) -> dict:
    """Compare technical signals across multiple stock symbols.

    Args:
        symbols: List of ticker symbols to compare.
        period: Lookback period supported by the data provider.
        interval: Candle interval.
        benchmark: Benchmark ticker for relative strength, default SPY.
    """
    return technical.compare_technical_signals(
        symbols=symbols,
        period=period,
        interval=interval,
        benchmark=benchmark,
    )


@mcp.tool()
def compare_technical_models(horizon_days: int | None = None) -> dict:
    """Compare trained technical ML models using the latest training manifest.

    Returns accuracy, Brier score, precision, and recall per model. If horizon_days
    is omitted, metrics for all trained horizons are returned.

    Args:
        horizon_days: Optional prediction horizon in trading days (e.g. 1, 5, 20).
    """
    return technical.compare_technical_models_tool(horizon_days=horizon_days)


@mcp.tool()
def predict_technical_direction(
    symbol: str,
    model: str = "logistic",
    horizon_days: int = 5,
) -> dict:
    """Predict up/down probability from a trained technical ML model.

    Supported models: baseline, logistic, random_forest, xgboost.
    Requires prior offline training via tradingai-train-models.

    Args:
        symbol: Ticker symbol, e.g. AAPL.
        model: Model name (baseline, logistic, random_forest, xgboost).
        horizon_days: Prediction horizon in trading days.
    """
    return technical.predict_technical_direction_tool(
        symbol=symbol,
        model=model,
        horizon_days=horizon_days,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
