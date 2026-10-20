"""MCP tool wrappers for IBKR trade / fill history."""

from __future__ import annotations

from tradingai.intraday.config import (
    get_ibkr_host,
    get_ibkr_mcp_client_id,
    get_ibkr_port,
)
from tradingai.providers.ibkr_tws import fetch_recent_trades


def get_last_trades_tool(
    limit: int = 20,
    side: str = "all",
    symbol: str | None = None,
    account: str | None = None,
) -> dict:
    """Return the most recent IBKR fills from TWS.

    Args:
        limit: Maximum number of trades to return (newest first).
        side: ``buy``, ``sell``, or ``all``.
        symbol: Optional ticker filter (e.g. AAPL).
        account: Optional IB account code filter.
    """
    try:
        return fetch_recent_trades(
            limit=limit,
            side=side,
            symbol=symbol,
            account=account,
            client_id=get_ibkr_mcp_client_id(),
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "count": 0,
            "limit": limit,
            "side": side,
            "symbol": symbol,
            "account": account,
            "trades": [],
            "host": get_ibkr_host(),
            "port": get_ibkr_port(),
            "hint": (
                "Ensure TWS/IB Gateway is running with API enabled "
                f"({get_ibkr_host()}:{get_ibkr_port()})."
            ),
            "disclaimer": (
                "IBKR TWS executions. Informational only — not financial advice."
            ),
        }
