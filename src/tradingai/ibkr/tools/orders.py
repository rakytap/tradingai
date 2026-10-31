"""MCP tool wrappers for IBKR equity buy/sell orders."""

from __future__ import annotations

from tradingai.intraday.config import (
    DEFAULT_BUY_NOTIONAL_USD,
    get_buy_notional_usd,
    get_ibkr_allow_orders,
    get_ibkr_host,
    get_ibkr_max_order_shares,
    get_ibkr_mcp_client_id,
    get_ibkr_port,
)
from tradingai.providers.ibkr_tws import (
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_BUY_STOP_LOSS,
    DEFAULT_SELL_TRAIL_PERCENT,
    place_budget_limit_buy,
    place_equity_order,
    place_limit_sell,
    place_trail_limit_sell,
)


def _place(
    *,
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "MKT",
    limit_price: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    try:
        return place_equity_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            tif=tif,
            outside_rth=outside_rth,
            account=account,
            dry_run=dry_run,
            allow_orders=get_ibkr_allow_orders(),
            max_shares=get_ibkr_max_order_shares(),
            client_id=get_ibkr_mcp_client_id(),
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "dry_run": dry_run,
            "submitted": False,
            "host": get_ibkr_host(),
            "port": get_ibkr_port(),
            "allow_orders": get_ibkr_allow_orders(),
            "hint": (
                "Ensure TWS/IB Gateway is running with API enabled "
                f"({get_ibkr_host()}:{get_ibkr_port()}). "
                "Live orders need TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false."
            ),
            "disclaimer": (
                "IBKR order helper. Real money risk if not on paper. "
                "Not financial advice."
            ),
        }


def buy_stock_tool(
    symbol: str,
    quantity: float,
    order_type: str = "MKT",
    limit_price: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Buy equity shares via IBKR TWS (dry-run by default)."""
    return _place(
        symbol=symbol,
        side="buy",
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        dry_run=dry_run,
    )


def sell_stock_tool(
    symbol: str,
    quantity: float,
    order_type: str = "MKT",
    limit_price: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Sell equity shares via IBKR TWS (dry-run by default)."""
    return _place(
        symbol=symbol,
        side="sell",
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        dry_run=dry_run,
    )


def buy_stock_budget_tool(
    symbol: str,
    total_budget: float | None = None,
    limit_price: float | None = None,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    stop_loss: float | None = DEFAULT_BUY_STOP_LOSS,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Limit-buy integer shares sized to ``budget_fraction`` of ``total_budget``.

    Defaults: LIMIT at current mid; spend 100% of total budget; no stop-loss;
    dry-run. Pass ``stop_loss`` (e.g. ``0.01``) to attach a SELL STP child.
    """
    budget = (
        float(total_budget)
        if total_budget is not None
        else float(get_buy_notional_usd() or DEFAULT_BUY_NOTIONAL_USD)
    )
    try:
        return place_budget_limit_buy(
            symbol=symbol,
            total_budget=budget,
            limit_price=limit_price,
            budget_fraction=budget_fraction,
            stop_loss=stop_loss,
            tif=tif,
            outside_rth=outside_rth,
            account=account,
            dry_run=dry_run,
            allow_orders=get_ibkr_allow_orders(),
            max_shares=get_ibkr_max_order_shares(),
            client_id=get_ibkr_mcp_client_id(),
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "symbol": symbol,
            "total_budget": budget,
            "limit_price": limit_price,
            "budget_fraction": budget_fraction,
            "stop_loss": stop_loss,
            "order_type": "LMT",
            "side": "buy",
            "dry_run": dry_run,
            "submitted": False,
            "host": get_ibkr_host(),
            "port": get_ibkr_port(),
            "allow_orders": get_ibkr_allow_orders(),
            "hint": (
                "Ensure TWS/IB Gateway is running with API enabled "
                f"({get_ibkr_host()}:{get_ibkr_port()}). "
                "Live orders need TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false. "
                "Mid price requires IB market data permissions."
            ),
            "disclaimer": (
                "IBKR order helper. Real money risk if not on paper. "
                "Not financial advice."
            ),
        }


def sell_stock_limit_tool(
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """LIMIT sell; defaults to full position and fee-aware target limit.

    ``target_return`` is a net-return fraction after fees (e.g. ``0.01`` = 1%).
    Used only when ``limit_price`` is omitted. Default =
    ``TRADINGAI_OPEN_DIP_TARGET_RETURN`` / ``DEFAULT_TARGET_RETURN``.
    """
    try:
        return place_limit_sell(
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            target_return=target_return,
            tif=tif,
            outside_rth=outside_rth,
            account=account,
            dry_run=dry_run,
            allow_orders=get_ibkr_allow_orders(),
            max_shares=get_ibkr_max_order_shares(),
            client_id=get_ibkr_mcp_client_id(),
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "symbol": symbol,
            "quantity": quantity,
            "limit_price": limit_price,
            "target_return": target_return,
            "order_type": "LMT",
            "side": "sell",
            "dry_run": dry_run,
            "submitted": False,
            "host": get_ibkr_host(),
            "port": get_ibkr_port(),
            "allow_orders": get_ibkr_allow_orders(),
            "hint": (
                "Ensure TWS/IB Gateway is running with API enabled "
                f"({get_ibkr_host()}:{get_ibkr_port()}). "
                "Live orders need TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false."
            ),
            "disclaimer": (
                "IBKR order helper. Real money risk if not on paper. "
                "Not financial advice."
            ),
        }


def sell_stock_trail_tool(
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    trail_percent: float = DEFAULT_SELL_TRAIL_PERCENT,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
) -> dict:
    """TRAIL LIMIT sell; defaults to full position and fee-aware target limit.

    ``target_return`` is a net-return fraction after fees (e.g. ``0.01`` = 1%).
    Used only when ``limit_price`` is omitted. Default =
    ``TRADINGAI_OPEN_DIP_TARGET_RETURN`` / ``DEFAULT_TARGET_RETURN``.
    """
    try:
        return place_trail_limit_sell(
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            trail_percent=trail_percent,
            target_return=target_return,
            tif=tif,
            outside_rth=outside_rth,
            account=account,
            dry_run=dry_run,
            allow_orders=get_ibkr_allow_orders(),
            max_shares=get_ibkr_max_order_shares(),
            client_id=get_ibkr_mcp_client_id(),
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "symbol": symbol,
            "quantity": quantity,
            "limit_price": limit_price,
            "trail_percent": trail_percent,
            "target_return": target_return,
            "order_type": "TRAIL LIMIT",
            "side": "sell",
            "dry_run": dry_run,
            "submitted": False,
            "host": get_ibkr_host(),
            "port": get_ibkr_port(),
            "allow_orders": get_ibkr_allow_orders(),
            "hint": (
                "Ensure TWS/IB Gateway is running with API enabled "
                f"({get_ibkr_host()}:{get_ibkr_port()}). "
                "Live orders need TRADINGAI_IBKR_ALLOW_ORDERS=1 and dry_run=false. "
                "Default qty needs an open position; default limit needs last buy "
                "or position avg cost."
            ),
            "disclaimer": (
                "IBKR order helper. Real money risk if not on paper. "
                "Not financial advice."
            ),
        }
