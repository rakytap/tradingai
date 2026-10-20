from mcp.server.fastmcp import FastMCP

from tradingai.ibkr.tools import orders, trades

mcp = FastMCP(
    "tradingai-ibkr",
    instructions=(
        "IBKR TWS tools for account executions and equity order placement "
        "(buy/sell). Requires TWS or IB Gateway with API enabled. "
        "Orders default to dry_run; live placement needs "
        "TRADINGAI_IBKR_ALLOW_ORDERS=1. Prefer paper port 7497. "
        "Outputs are informational only and not financial advice."
    ),
)


@mcp.tool()
def get_last_trades(
    limit: int = 20,
    side: str = "all",
    symbol: str | None = None,
    account: str | None = None,
) -> dict:
    """Download recent IBKR trade fills from TWS (newest first).

    Uses the TWS API ``reqExecutions`` endpoint. IB typically returns
    executions from the current session / recent trading day only — not
    full multi-year history.

    Args:
        limit: Number of trades to return after filtering (1–500, default 20).
        side: Filter by direction: ``buy``, ``sell``, or ``all`` (default).
        symbol: Optional ticker filter, e.g. AAPL.
        account: Optional IB account code (blank = all visible accounts).
    """
    return trades.get_last_trades_tool(
        limit=limit,
        side=side,
        symbol=symbol,
        account=account,
    )


@mcp.tool()
def buy_stock(
    symbol: str,
    quantity: float,
    order_type: str = "MKT",
    limit_price: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Buy a US equity via IBKR TWS (SMART/USD).

    Defaults to dry_run=true (validates and previews only). To submit a live
    order: set TRADINGAI_IBKR_ALLOW_ORDERS=1 in .env and pass dry_run=false.
    Prefer paper TWS port 7497 while testing.

    Args:
        symbol: Ticker, e.g. AAPL.
        quantity: Shares to buy (must be <= TRADINGAI_IBKR_MAX_ORDER_SHARES).
        order_type: MKT (default) or LMT.
        limit_price: Required when order_type=LMT.
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        dry_run: If true (default), do not send the order to TWS.
    """
    return orders.buy_stock_tool(
        symbol=symbol,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        dry_run=dry_run,
    )


@mcp.tool()
def sell_stock(
    symbol: str,
    quantity: float,
    order_type: str = "MKT",
    limit_price: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Sell a US equity via IBKR TWS (SMART/USD).

    Defaults to dry_run=true (validates and previews only). To submit a live
    order: set TRADINGAI_IBKR_ALLOW_ORDERS=1 in .env and pass dry_run=false.
    Prefer paper TWS port 7497 while testing.

    Args:
        symbol: Ticker, e.g. AAPL.
        quantity: Shares to sell (must be <= TRADINGAI_IBKR_MAX_ORDER_SHARES).
        order_type: MKT (default) or LMT.
        limit_price: Required when order_type=LMT.
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        dry_run: If true (default), do not send the order to TWS.
    """
    return orders.sell_stock_tool(
        symbol=symbol,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        dry_run=dry_run,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
