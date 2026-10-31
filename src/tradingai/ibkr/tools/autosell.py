"""MCP wrappers for the IBKR autosell background watcher."""

from __future__ import annotations

from tradingai.ibkr.autosell import (
    DEFAULT_POLL_INTERVAL_SEC,
    AutosellConfig,
    job_status,
    start_background_job,
    stop_job,
)
from tradingai.providers.ibkr_tws import DEFAULT_SELL_TRAIL_PERCENT


def autosell_watch_start_tool(
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    trail_percent: float = DEFAULT_SELL_TRAIL_PERCENT,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
) -> dict:
    """Start a background autosell watcher (same sell args as tradingai-sell-trail)."""
    try:
        config = AutosellConfig(
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            trail_percent=trail_percent,
            target_return=target_return,
            tif=tif,
            outside_rth=outside_rth,
            account=account,
            submit=bool(submit),
            interval_sec=float(interval_sec),
        )
        return start_background_job(config)
    except Exception as exc:
        return {
            "error": str(exc),
            "symbol": symbol,
            "submit": bool(submit),
            "interval_sec": interval_sec,
            "disclaimer": (
                "IBKR autosell helper. Real money risk if submit=true. "
                "Not financial advice."
            ),
        }


def autosell_watch_status_tool(job_id: str | None = None) -> dict:
    try:
        return job_status(job_id)
    except Exception as exc:
        return {"error": str(exc), "job_id": job_id}


def autosell_watch_stop_tool(job_id: str, force: bool = False) -> dict:
    try:
        return stop_job(job_id, force=bool(force))
    except Exception as exc:
        return {"error": str(exc), "job_id": job_id}
