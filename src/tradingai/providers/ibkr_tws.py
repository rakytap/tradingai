"""IBKR TWS historical bars for tradingai-sync-intraday (ib_insync).

Not registered as the global TRADINGAI_DATA_PROVIDER — sync-only.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from tradingai.intraday.bars import normalize_intraday_index
from tradingai.intraday.config import (
    get_bar_interval,
    get_bar_period,
    get_ibkr_client_id,
    get_ibkr_host,
    get_ibkr_port,
)

_PERIOD_RE = re.compile(r"^(\d+)\s*([a-zA-Z]+)$")


def period_to_ib_duration(period: str | None = None) -> str:
    """Map yfinance-style period (e.g. 5d) to IB durationStr (e.g. 5 D)."""
    raw = (period or get_bar_period()).strip().lower()
    match = _PERIOD_RE.match(raw)
    if not match:
        raise ValueError(
            f"Unsupported TRADINGAI_INTRADAY_PERIOD '{raw}'. "
            "Use forms like 1d, 5d, 1w (IB durationStr)."
        )
    count = int(match.group(1))
    unit = match.group(2)
    unit_map = {
        "d": "D",
        "day": "D",
        "days": "D",
        "w": "W",
        "week": "W",
        "weeks": "W",
        "m": "M",
        "mo": "M",
        "month": "M",
        "months": "M",
        "y": "Y",
        "year": "Y",
        "years": "Y",
    }
    ib_unit = unit_map.get(unit)
    if ib_unit is None:
        raise ValueError(
            f"Unsupported period unit in '{raw}'. Use d/w/m/y (e.g. 5d)."
        )
    if count < 1:
        raise ValueError(f"Period count must be >= 1, got {count} from '{raw}'.")
    return f"{count} {ib_unit}"


def interval_to_ib_bar_size(interval: str | None = None) -> str:
    """Map TRADINGAI_INTRADAY_INTERVAL to IB barSizeSetting."""
    raw = (interval or get_bar_interval()).strip().lower()
    mapping = {
        "1m": "1 min",
        "1min": "1 min",
        "1minute": "1 min",
        "5m": "5 mins",
        "5min": "5 mins",
        "15m": "15 mins",
        "15min": "15 mins",
        "30m": "30 mins",
        "30min": "30 mins",
        "1h": "1 hour",
        "1hour": "1 hour",
    }
    bar_size = mapping.get(raw)
    if bar_size is None:
        raise ValueError(
            f"Unsupported TRADINGAI_INTRADAY_INTERVAL '{raw}' for IBKR sync. "
            "Supported: 1m, 5m, 15m, 30m, 1h."
        )
    return bar_size


@contextmanager
def connect_ib(
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> Iterator:
    """Connect to TWS / IB Gateway; disconnect on exit."""
    from ib_insync import IB

    ib = IB()
    connect_host = host if host is not None else get_ibkr_host()
    connect_port = port if port is not None else get_ibkr_port()
    connect_client_id = client_id if client_id is not None else get_ibkr_client_id()
    try:
        ib.connect(connect_host, connect_port, clientId=connect_client_id, readonly=True)
        yield ib
    finally:
        if ib.isConnected():
            ib.disconnect()


def bars_to_dataframe(bars) -> pd.DataFrame:
    """Convert ib_insync BarDataList to normalized OHLCV DataFrame."""
    if not bars:
        raise ValueError("No OHLCV data returned for the requested symbol and range.")
    rows = []
    index = []
    for bar in bars:
        index.append(pd.Timestamp(bar.date))
        rows.append(
            {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume) if bar.volume is not None else 0.0,
            }
        )
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index))
    return normalize_intraday_index(frame)


def fetch_intraday_ohlcv_ibkr(
    ib,
    ticker: str,
    *,
    period: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV for one equity via an open IB connection."""
    from ib_insync import Stock

    symbol = ticker.upper().strip()
    duration = period_to_ib_duration(period)
    bar_size = interval_to_ib_bar_size(interval)
    contract = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise ValueError(f"IBKR could not qualify contract for {symbol}.")
    bars = ib.reqHistoricalData(
        qualified[0],
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    return bars_to_dataframe(bars)
