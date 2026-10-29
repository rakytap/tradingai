from mcp.server.fastmcp import FastMCP

from tradingai.ibkr.tools import orders, trades

mcp = FastMCP(
    "tradingai-ibkr",
    instructions=(
        "IBKR TWS tools for account executions and equity order placement "
        "(buy/sell). Prefer buy_stock_budget for budget limit buys and "
        "sell_stock_trail for TRAIL LIMIT sells (default: full position, "
        "fee-aware target limit, 0.05% trail). "
        "Requires TWS or IB Gateway with API enabled. "
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
def buy_stock_budget(
    symbol: str,
    total_budget: float | None = None,
    limit_price: float | None = None,
    budget_fraction: float = 0.95,
    stop_loss: float | None = 0.01,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Buy a US equity with a LIMIT order sized to a cash budget.

    Computes integer shares as floor((total_budget * budget_fraction) /
    limit_price). Default limit_price is the current IB mid; default
    budget_fraction is 0.95; default total_budget is TRADINGAI_OPEN_DIP_BUY_NOTIONAL
    (or 20000). Default stop_loss is 0.01 (1% SELL STP child under the buy
    limit). Pass stop_loss=0 to skip. Dry-run by default — live needs
    TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false.

    Args:
        symbol: Ticker, e.g. AAPL.
        total_budget: Gross cash budget in USD (spend = budget * fraction).
        limit_price: Limit price; omit to use current mid from TWS.
        budget_fraction: Fraction of total_budget to spend (default 0.95).
        stop_loss: Stop-loss fraction under limit (default 0.01 = 1%); 0 skips.
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        dry_run: If true (default), do not send the order to TWS.
    """
    return orders.buy_stock_budget_tool(
        symbol=symbol,
        total_budget=total_budget,
        limit_price=limit_price,
        budget_fraction=budget_fraction,
        stop_loss=stop_loss,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        dry_run=dry_run,
    )


@mcp.tool()
def sell_stock_trail(
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    trail_percent: float = 0.05,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Sell a US equity with a TRAIL LIMIT order via IBKR TWS.

    Defaults: full open long position; limit_price = fee-aware target at
    TRADINGAI_OPEN_DIP_TARGET_RETURN (≈0.3836% net) on last buy / avg cost;
    trail 0.05%. Pass target_return as a fraction (0.01 = 1% net after fees)
    when limit_price is omitted. Dry-run by default — live needs
    TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false.

    Args:
        symbol: Ticker, e.g. AAPL.
        quantity: Shares to sell; omit to sell the full open position.
        limit_price: Trail stop / limit reference; omit for fee-aware target.
        trail_percent: IB trailing percent points (0.05 = 0.05%).
        target_return: Net-return fraction after fees (e.g. 0.01 = 1%);
            used only when limit_price is omitted.
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        dry_run: If true (default), do not send the order to TWS.
    """
    return orders.sell_stock_trail_tool(
        symbol=symbol,
        quantity=quantity,
        limit_price=limit_price,
        trail_percent=trail_percent,
        target_return=target_return,
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
