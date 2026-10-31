"""MCP wrappers for sell-trail watcher (activate at profit, stepped trail)."""

from __future__ import annotations

from tradingai.ibkr.sell_trail import (
    DEFAULT_POLL_INTERVAL_SEC,
    job_status,
    start_sell_trail,
    stop_job,
)
from tradingai.providers.ibkr_tws import (
    DEFAULT_SELL_TRAIL_1_5X_PERCENT,
    DEFAULT_SELL_TRAIL_2X_PERCENT,
    DEFAULT_SELL_TRAIL_ACTIVATE_RETURN,
    DEFAULT_SELL_TRAIL_PERCENT,
)


def sell_trail_start_tool(
    symbol: str,
    quantity: float | None = None,
    buy_price: float | None = None,
    activate_return: float = DEFAULT_SELL_TRAIL_ACTIVATE_RETURN,
    initial_trail_percent: float = DEFAULT_SELL_TRAIL_PERCENT,
    trail_1_5x_percent: float = DEFAULT_SELL_TRAIL_1_5X_PERCENT,
    trail_2x_percent: float = DEFAULT_SELL_TRAIL_2X_PERCENT,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
) -> dict:
    try:
        return start_sell_trail(
            symbol=symbol,
            quantity=quantity,
            buy_price=buy_price,
            activate_return=float(activate_return),
            initial_trail_percent=float(initial_trail_percent),
            trail_1_5x_percent=float(trail_1_5x_percent),
            trail_2x_percent=float(trail_2x_percent),
            tif=tif,
            outside_rth=outside_rth,
            account=account,
            submit=bool(submit),
            interval_sec=float(interval_sec),
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "symbol": symbol,
            "submit": bool(submit),
            "disclaimer": (
                "IBKR sell-trail helper. Real money risk if submit=true. "
                "Not financial advice."
            ),
        }


def sell_trail_status_tool(job_id: str | None = None) -> dict:
    try:
        return job_status(job_id)
    except Exception as exc:
        return {"error": str(exc), "job_id": job_id}


def sell_trail_stop_tool(job_id: str, force: bool = False) -> dict:
    try:
        return stop_job(job_id, force=bool(force))
    except Exception as exc:
        return {"error": str(exc), "job_id": job_id}
