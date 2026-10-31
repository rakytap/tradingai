"""MCP wrappers for budget-buy + protective trail watcher."""

from __future__ import annotations

from tradingai.ibkr.buy_trail import (
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_TIGHT_POLL_INTERVAL_SEC,
    job_status,
    start_buy_trail,
    stop_job,
)
from tradingai.providers.ibkr_tws import (
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_BUY_TRAIL_1_5X_PERCENT,
    DEFAULT_BUY_TRAIL_2X_PERCENT,
    DEFAULT_BUY_TRAIL_PERCENT,
    DEFAULT_BUY_TRAIL_TIGHT_PERCENT,
)


def buy_trail_start_tool(
    symbol: str,
    total_budget: float | None = None,
    limit_price: float | None = None,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    initial_trail_percent: float = DEFAULT_BUY_TRAIL_PERCENT,
    tight_trail_percent: float = DEFAULT_BUY_TRAIL_TIGHT_PERCENT,
    trail_1_5x_percent: float = DEFAULT_BUY_TRAIL_1_5X_PERCENT,
    trail_2x_percent: float = DEFAULT_BUY_TRAIL_2X_PERCENT,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    tight_interval_sec: float = DEFAULT_TIGHT_POLL_INTERVAL_SEC,
) -> dict:
    try:
        return start_buy_trail(
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
            submit=bool(submit),
            interval_sec=float(interval_sec),
            tight_interval_sec=float(tight_interval_sec),
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "symbol": symbol,
            "submit": bool(submit),
            "disclaimer": (
                "IBKR buy-trail helper. Real money risk if submit=true. "
                "Not financial advice."
            ),
        }


def buy_trail_status_tool(job_id: str | None = None) -> dict:
    try:
        return job_status(job_id)
    except Exception as exc:
        return {"error": str(exc), "job_id": job_id}


def buy_trail_stop_tool(job_id: str, force: bool = False) -> dict:
    try:
        return stop_job(job_id, force=bool(force))
    except Exception as exc:
        return {"error": str(exc), "job_id": job_id}
