from mcp.server.fastmcp import FastMCP

from tradingai.regime.tools import regime

mcp = FastMCP(
    "tradingai-regime",
    instructions=(
        "Market regime detection tools for structured L1/L4 regime state "
        "(market factors + sector board). Outputs are informational only and not financial advice."
    ),
)


@mcp.tool()
def fetch_macro_features() -> dict:
    """Fetch macro feature values used for market regime detection.

    Includes VIX, SPY trend, yield curve, rates, inflation, and credit spread
    indicators from enabled providers.
    """
    return regime.fetch_macro_features_tool()


@mcp.tool()
def detect_market_regime() -> dict:
    """Detect the current market regime label (SPY trend + VIX) and strategy.

    Prefer get_regime_state for the full structured L1/L4 board.
    """
    return regime.detect_market_regime()


@mcp.tool()
def get_regime_state(persist: bool = False) -> dict:
    """Return structured regime state: market factors + per-sector board.

    Market factors include risk, vol, rates, credit, growth_value, equity_trend.
    Sector board maps each GICS sector to bullish/neutral/bearish via sector ETFs.
    Set persist=true to write a regime_snapshots row.
    """
    return regime.get_regime_state(persist=persist)


@mcp.tool()
def explain_regime() -> dict:
    """Return a human-readable explanation of the current market regime label."""
    return regime.explain_market_regime()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
