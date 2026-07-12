from mcp.server.fastmcp import FastMCP

from tradingai.regime.tools import regime

mcp = FastMCP(
    "tradingai-regime",
    instructions=(
        "Market regime detection tools for bull, bear, sideways, and high-volatility "
        "environments. Outputs are informational only and not financial advice."
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
    """Detect the current market regime and recommended strategy.

    Returns regime label, recommended_strategy, confidence, features, and signals.
    This is informational macro analysis, not financial advice.
    """
    return regime.detect_market_regime()


@mcp.tool()
def explain_regime() -> dict:
    """Return a human-readable explanation of the current market regime."""
    return regime.explain_market_regime()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
