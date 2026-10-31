from mcp.server.fastmcp import FastMCP

from tradingai.ibkr.tools import autosell, buy_trail, orders, sell_trail, trades

mcp = FastMCP(
    "tradingai-ibkr",
    instructions=(
        "IBKR TWS tools for account executions and equity order placement "
        "(buy/sell). Prefer buy_stock_budget for budget limit buys and "
        "sell_stock_limit for LIMIT sells (default: full position, "
        "fee-aware target limit). Use sell_stock_trail for a one-shot "
        "TRAIL LIMIT (default 0.05% trail; no watcher / no trail updates). "
        "Use buy_trail_start for a budget LIMIT buy (mid default, no STP) "
        "then background trail: 0.6% after fill; tighten to 0.05%/0.03%/0.01% "
        "at 1×/1.5×/2× fee-aware target (poll 5s after first tighten); "
        "exit when position is flat. "
        "Use sell_trail_start only for the stepped watcher: wait for "
        "fee-aware 0.2% profit then TRAIL 0.05%; tighten to 0.03%/0.01% at "
        "1.5×/2× activation profit (poll 10s); exit when sold or final "
        "tighten done. Use autosell_watch_start to monitor unpaired buys. "
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
    budget_fraction: float = 1.0,
    stop_loss: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Buy a US equity with a LIMIT order sized to a cash budget.

    Computes integer shares as floor((total_budget * budget_fraction) /
    limit_price). Default limit_price is the current IB mid; default
    budget_fraction is 1.0; default total_budget is TRADINGAI_OPEN_DIP_BUY_NOTIONAL
    (or 20000). No stop-loss by default — pass stop_loss (e.g. 0.01) for a
    SELL STP child under the buy limit. Dry-run by default — live needs
    TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false.

    Args:
        symbol: Ticker, e.g. AAPL.
        total_budget: Gross cash budget in USD (spend = budget * fraction).
        limit_price: Limit price; omit to use current mid from TWS.
        budget_fraction: Fraction of total_budget to spend (default 1.0).
        stop_loss: Stop-loss fraction under limit (default None = no stop);
            e.g. 0.01 for 1% STP child.
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
def sell_stock_limit(
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Sell a US equity with a LIMIT order via IBKR TWS.

    Defaults: full open long position; limit_price = fee-aware target at
    TRADINGAI_OPEN_DIP_TARGET_RETURN on last buy / avg cost. Pass
    target_return as a fraction (0.01 = 1% net after fees) when limit_price
    is omitted. Dry-run by default — live needs
    TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false.

    Args:
        symbol: Ticker, e.g. AAPL.
        quantity: Shares to sell; omit to sell the full open position.
        limit_price: Limit price; omit for fee-aware target.
        target_return: Net-return fraction after fees (e.g. 0.01 = 1%);
            used only when limit_price is omitted.
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        dry_run: If true (default), do not send the order to TWS.
    """
    return orders.sell_stock_limit_tool(
        symbol=symbol,
        quantity=quantity,
        limit_price=limit_price,
        target_return=target_return,
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
    """One-shot TRAIL LIMIT sell via IBKR TWS (no watcher / no trail updates).

    Prefer sell_stock_limit for a plain LIMIT sell. Prefer sell_trail_start only
    when a stepped profit watcher is explicitly requested. Defaults: full open
    long position; limit_price = fee-aware target at
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
def autosell_watch_start(
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    trail_percent: float = 0.05,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = 15.0,
) -> dict:
    """Start a background job that autosells unpaired IBKR buys.

    Polls TWS executions every ``interval_sec`` (default 15). When a buy fill
    has no matching sell (FIFO), places a TRAIL LIMIT sell for that buy's share
    count using the same arguments as ``sell_stock_trail`` /
    ``tradingai-sell-trail``.
    ``submit=false`` (default) dry-runs sells; live needs
    TRADINGAI_IBKR_ALLOW_ORDERS=1 and submit=true.

    Args:
        symbol: Ticker to watch, e.g. COIN.
        quantity: Optional max shares per autosell (default = exact buy qty).
        limit_price: Trail stop reference; omit for fee-aware target on the buy.
        trail_percent: IB trailing percent points (0.05 = 0.05%).
        target_return: Net-return fraction after fees when limit_price omitted.
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        submit: If true, send live sells; default false (dry-run).
        interval_sec: Poll interval in seconds (default 15).
    """
    return autosell.autosell_watch_start_tool(
        symbol=symbol,
        quantity=quantity,
        limit_price=limit_price,
        trail_percent=trail_percent,
        target_return=target_return,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        submit=submit,
        interval_sec=interval_sec,
    )


@mcp.tool()
def autosell_watch_status(job_id: str | None = None) -> dict:
    """Status for one autosell job, or a list of all jobs when job_id omitted."""
    return autosell.autosell_watch_status_tool(job_id=job_id)


@mcp.tool()
def autosell_watch_stop(job_id: str, force: bool = False) -> dict:
    """Stop a background autosell watcher by job_id."""
    return autosell.autosell_watch_stop_tool(job_id=job_id, force=force)


@mcp.tool()
def buy_trail_start(
    symbol: str,
    total_budget: float | None = None,
    limit_price: float | None = None,
    budget_fraction: float = 1.0,
    initial_trail_percent: float = 0.6,
    tight_trail_percent: float = 0.05,
    trail_1_5x_percent: float = 0.03,
    trail_2x_percent: float = 0.01,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = 10.0,
    tight_interval_sec: float = 5.0,
) -> dict:
    """Budget LIMIT buy (no STP), then stepped trail after fill.

    Default buy limit is current IB mid (trail activate). No STP child. On
    ``submit=true``, a background job polls every ``interval_sec`` (default 10)
    until the first tighten, then every ``tight_interval_sec`` (default 5):
    wait for full fill → TRAIL LIMIT at ``initial_trail_percent`` (0.6%);
    at fee-aware 1× / 1.5× / 2× ``target_return`` tighten to 0.05% / 0.03% /
    0.01%. Job exits when the long position is flat. Dry-run by default.

    Args:
        symbol: Ticker, e.g. COIN.
        total_budget: Gross cash budget USD.
        limit_price: Buy limit / trail activate; omit for current mid.
        budget_fraction: Fraction of budget to spend (default 1.0).
        initial_trail_percent: IB trailing percent after fill (default 0.6).
        tight_trail_percent: Trail at 1× target mid (default 0.05).
        trail_1_5x_percent: Trail at 1.5× target mid (default 0.03).
        trail_2x_percent: Trail at 2× target mid (default 0.01).
        target_return: Net-return fraction for 1× level (default env / ~0.3836%).
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        submit: Live buy + live trail watcher if true.
        interval_sec: Poll interval before first tighten (default 10).
        tight_interval_sec: Poll interval after 0.05% trail (default 5).
    """
    return buy_trail.buy_trail_start_tool(
        symbol=symbol,
        total_budget=total_budget,
        limit_price=limit_price,
        budget_fraction=budget_fraction,
        initial_trail_percent=initial_trail_percent,
        tight_trail_percent=tight_trail_percent,
        trail_1_5x_percent=trail_1_5x_percent,
        trail_2x_percent=trail_2x_percent,
        target_return=target_return,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        submit=submit,
        interval_sec=interval_sec,
        tight_interval_sec=tight_interval_sec,
    )


@mcp.tool()
def buy_trail_status(job_id: str | None = None) -> dict:
    """Status for one buy-trail job, or list all jobs."""
    return buy_trail.buy_trail_status_tool(job_id=job_id)


@mcp.tool()
def buy_trail_stop(job_id: str, force: bool = False) -> dict:
    """Stop a buy-trail background watcher by job_id."""
    return buy_trail.buy_trail_stop_tool(job_id=job_id, force=force)


@mcp.tool()
def sell_trail_start(
    symbol: str,
    quantity: float | None = None,
    buy_price: float | None = None,
    activate_return: float = 0.002,
    initial_trail_percent: float = 0.05,
    trail_1_5x_percent: float = 0.03,
    trail_2x_percent: float = 0.01,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = 10.0,
) -> dict:
    """Stepped trail watcher on an open long (not a one-shot TRAIL LIMIT).

    Prefer ``sell_stock_trail`` / ``tradingai-sell-trail`` for a simple TRAIL
    LIMIT with no background job. This tool resolves buy price from IBKR
    (last buy fill, else avg cost), polls mid every ``interval_sec`` (default
    10), arms TRAIL at fee-aware ``activate_return`` (default 0.2%) with
    ``initial_trail_percent`` (0.05%), then tightens to 0.03% / 0.01% at
    1.5× / 2× activation profit. Watcher exits when flat or after the final
    tighten. Dry-run by default. CLI: ``tradingai-sell-trail-watch``.

    Args:
        symbol: Ticker, e.g. COIN.
        quantity: Shares to trail (default full open long).
        buy_price: Override buy reference; omit to query IBKR.
        activate_return: Fee-aware net return that arms the trail (default 0.002).
        initial_trail_percent: IB trail at activation (default 0.05).
        trail_1_5x_percent: Trail at 1.5× activation profit (default 0.03).
        trail_2x_percent: Trail at 2× activation profit (default 0.01).
        tif: Time in force, default DAY.
        outside_rth: Allow outside regular trading hours.
        account: Optional IB account code.
        submit: Live watcher if true.
        interval_sec: Poll interval seconds (default 10).
    """
    return sell_trail.sell_trail_start_tool(
        symbol=symbol,
        quantity=quantity,
        buy_price=buy_price,
        activate_return=activate_return,
        initial_trail_percent=initial_trail_percent,
        trail_1_5x_percent=trail_1_5x_percent,
        trail_2x_percent=trail_2x_percent,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        submit=submit,
        interval_sec=interval_sec,
    )


@mcp.tool()
def sell_trail_status(job_id: str | None = None) -> dict:
    """Status for one sell-trail job, or list all jobs."""
    return sell_trail.sell_trail_status_tool(job_id=job_id)


@mcp.tool()
def sell_trail_stop(job_id: str, force: bool = False) -> dict:
    """Stop a sell-trail background watcher by job_id."""
    return sell_trail.sell_trail_stop_tool(job_id=job_id, force=force)


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
