"""MCP tool wrappers for IBKR equity buy/sell orders."""

from __future__ import annotations

from tradingai.intraday.config import (
    get_ibkr_allow_orders,
    get_ibkr_host,
    get_ibkr_max_order_shares,
    get_ibkr_mcp_client_id,
    get_ibkr_port,
)
from tradingai.providers.ibkr_tws import place_equity_order


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
