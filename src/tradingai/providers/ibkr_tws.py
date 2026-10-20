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
    readonly: bool = True,
) -> Iterator:
    """Connect to TWS / IB Gateway; disconnect on exit.

    Use ``readonly=False`` only for order placement (buy/sell).
    """
    from ib_insync import IB

    ib = IB()
    connect_host = host if host is not None else get_ibkr_host()
    connect_port = port if port is not None else get_ibkr_port()
    connect_client_id = client_id if client_id is not None else get_ibkr_client_id()
    try:
        ib.connect(
            connect_host,
            connect_port,
            clientId=connect_client_id,
            readonly=readonly,
        )
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
    # IBKR share classes use a space (BRK B); Yahoo-style files use BRK.B / BRK-B.
    ib_symbol = symbol.replace(".", " ").replace("-", " ")
    duration = period_to_ib_duration(period)
    bar_size = interval_to_ib_bar_size(interval)
    contract = Stock(ib_symbol, "SMART", "USD")
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


def normalize_trade_side(side: str | None) -> str | None:
    """Map buy/sell aliases to canonical 'buy' / 'sell'; None means no filter."""
    if side is None:
        return None
    raw = str(side).strip().lower()
    if raw in ("", "all", "any", "*"):
        return None
    if raw in ("buy", "bot", "b"):
        return "buy"
    if raw in ("sell", "sld", "s"):
        return "sell"
    raise ValueError(
        f"Unsupported side '{side}'. Use buy, sell, or all."
    )


def execution_side_to_canonical(side: str | None) -> str:
    """Normalize IB execution side (BOT/SLD/BUY/SELL) to buy/sell/unknown."""
    raw = (side or "").strip().upper()
    if raw in ("BUY", "BOT", "B"):
        return "buy"
    if raw in ("SELL", "SLD", "S"):
        return "sell"
    return "unknown"


def serialize_fill(fill) -> dict:
    """Convert an ib_insync Fill to a JSON-serializable dict."""
    execution = fill.execution
    contract = fill.contract
    commission = getattr(fill, "commissionReport", None)
    commission_usd = None
    currency = None
    realized_pnl = None
    if commission is not None:
        if getattr(commission, "commission", None) is not None:
            try:
                commission_usd = float(commission.commission)
            except (TypeError, ValueError):
                commission_usd = None
        currency = getattr(commission, "currency", None) or None
        if getattr(commission, "realizedPNL", None) is not None:
            try:
                # IB uses 1e100 as "unset"
                pnl = float(commission.realizedPNL)
                if abs(pnl) < 1e99:
                    realized_pnl = pnl
            except (TypeError, ValueError):
                realized_pnl = None
    trade_time = getattr(fill, "time", None) or getattr(execution, "time", None)
    shares = float(execution.shares) if execution.shares is not None else 0.0
    price = float(execution.price) if execution.price is not None else 0.0
    return {
        "exec_id": execution.execId,
        "time": trade_time.isoformat() if trade_time is not None else None,
        "symbol": getattr(contract, "symbol", None),
        "local_symbol": getattr(contract, "localSymbol", None),
        "sec_type": getattr(contract, "secType", None),
        "exchange": execution.exchange or getattr(contract, "exchange", None),
        "side": execution_side_to_canonical(execution.side),
        "side_raw": execution.side,
        "shares": shares,
        "price": price,
        "notional_usd": round(shares * price, 4),
        "avg_price": float(execution.avgPrice) if execution.avgPrice is not None else None,
        "cum_qty": float(execution.cumQty) if execution.cumQty is not None else None,
        "account": execution.acctNumber,
        "order_id": execution.orderId,
        "perm_id": execution.permId,
        "client_id": execution.clientId,
        "order_ref": execution.orderRef or None,
        "commission": commission_usd,
        "commission_currency": currency,
        "realized_pnl": realized_pnl,
    }


def fetch_recent_trades(
    *,
    limit: int = 20,
    side: str | None = None,
    symbol: str | None = None,
    account: str | None = None,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict:
    """Fetch recent IBKR fills via TWS ``reqExecutions``, newest first.

    IB typically returns executions from the current session / recent day only
    (not full account history). ``limit`` truncates after side/symbol filters.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if limit > 500:
        raise ValueError("limit must be <= 500")

    side_filter = normalize_trade_side(side)
    symbol_filter = symbol.upper().strip() if symbol else None
    account_filter = account.strip() if account else None

    from ib_insync import ExecutionFilter

    # IB ExecutionFilter.side expects BUY/SELL; also filter client-side.
    ib_side = ""
    if side_filter == "buy":
        ib_side = "BUY"
    elif side_filter == "sell":
        ib_side = "SELL"

    exec_filter = ExecutionFilter(
        acctCode=account_filter or "",
        symbol=symbol_filter or "",
        side=ib_side,
    )

    with connect_ib(host=host, port=port, client_id=client_id) as ib:
        fills = list(ib.reqExecutions(exec_filter) or [])

    trades = [serialize_fill(fill) for fill in fills]
    if side_filter is not None:
        trades = [t for t in trades if t["side"] == side_filter]
    if symbol_filter is not None:
        trades = [
            t
            for t in trades
            if (t.get("symbol") or "").upper() == symbol_filter
            or (t.get("local_symbol") or "").upper() == symbol_filter
        ]
    if account_filter is not None:
        trades = [t for t in trades if (t.get("account") or "") == account_filter]

    trades.sort(key=lambda t: t.get("time") or "", reverse=True)
    truncated = trades[:limit]
    return {
        "count": len(truncated),
        "limit": limit,
        "side": side_filter or "all",
        "symbol": symbol_filter,
        "account": account_filter,
        "available_before_limit": len(trades),
        "trades": truncated,
        "source": "ibkr_tws",
        "disclaimer": (
            "IBKR TWS executions (typically current session / recent day). "
            "Informational only — not financial advice."
        ),
    }


def _normalize_order_action(side: str) -> str:
    canonical = normalize_trade_side(side)
    if canonical == "buy":
        return "BUY"
    if canonical == "sell":
        return "SELL"
    raise ValueError("side must be buy or sell.")


def _normalize_order_type(order_type: str) -> str:
    raw = (order_type or "MKT").strip().upper()
    if raw in ("MKT", "MARKET"):
        return "MKT"
    if raw in ("LMT", "LIMIT"):
        return "LMT"
    raise ValueError("order_type must be MKT or LMT.")


def place_equity_order(
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
    allow_orders: bool = False,
    max_shares: float = 1_000.0,
    wait_seconds: float = 2.0,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict:
    """Place (or dry-run) a SMART/USD equity market or limit order via TWS.

    Live placement requires ``dry_run=False`` and ``allow_orders=True``.
    """
    from ib_insync import LimitOrder, MarketOrder, Stock

    ticker = symbol.upper().strip()
    if not ticker:
        raise ValueError("symbol is required.")
    qty = float(quantity)
    if qty <= 0.0:
        raise ValueError("quantity must be positive.")
    if max_shares > 0.0 and qty > max_shares:
        raise ValueError(
            f"quantity {qty} exceeds max_shares={max_shares}. "
            "Raise TRADINGAI_IBKR_MAX_ORDER_SHARES if intentional."
        )

    action = _normalize_order_action(side)
    otype = _normalize_order_type(order_type)
    if otype == "LMT":
        if limit_price is None or float(limit_price) <= 0.0:
            raise ValueError("limit_price is required for LMT orders.")
        limit_price = float(limit_price)

    tif_norm = (tif or "DAY").strip().upper()
    ib_symbol = ticker.replace(".", " ").replace("-", " ")

    preview = {
        "symbol": ticker,
        "side": action.lower(),
        "quantity": qty,
        "order_type": otype,
        "limit_price": limit_price,
        "tif": tif_norm,
        "outside_rth": bool(outside_rth),
        "account": account,
        "exchange": "SMART",
        "currency": "USD",
        "dry_run": bool(dry_run),
        "source": "ibkr_tws",
        "disclaimer": (
            "IBKR order helper. Real money risk if not on paper. "
            "Not financial advice."
        ),
    }

    if dry_run:
        preview["status"] = "dry_run"
        preview["submitted"] = False
        preview["message"] = (
            "Dry run only — order not sent. "
            "Call with dry_run=false and TRADINGAI_IBKR_ALLOW_ORDERS=1 to place."
        )
        return preview

    if not allow_orders:
        raise PermissionError(
            "Order placement blocked. Set TRADINGAI_IBKR_ALLOW_ORDERS=1 "
            "and pass dry_run=false to submit."
        )

    with connect_ib(
        host=host,
        port=port,
        client_id=client_id,
        readonly=False,
    ) as ib:
        contract = Stock(ib_symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"IBKR could not qualify contract for {ticker}.")
        contract = qualified[0]

        if otype == "MKT":
            order = MarketOrder(action, qty)
        else:
            order = LimitOrder(action, qty, limit_price)
        order.tif = tif_norm
        order.outsideRth = bool(outside_rth)
        if account:
            order.account = account.strip()

        trade = ib.placeOrder(contract, order)
        if wait_seconds > 0:
            ib.sleep(wait_seconds)

        order_status = trade.orderStatus
        fills = []
        for fill in trade.fills or []:
            fills.append(serialize_fill(fill))

        return {
            **preview,
            "status": order_status.status if order_status else "submitted",
            "submitted": True,
            "order_id": trade.order.orderId if trade.order else None,
            "perm_id": trade.order.permId if trade.order else None,
            "filled": float(order_status.filled) if order_status else 0.0,
            "remaining": float(order_status.remaining) if order_status else qty,
            "avg_fill_price": (
                float(order_status.avgFillPrice) if order_status else None
            ),
            "why_held": getattr(order_status, "whyHeld", None) or None,
            "log": [str(entry) for entry in (trade.log or [])][-8:],
            "fills": fills,
            "message": f"Order submitted to TWS ({order_status.status if order_status else 'unknown'}).",
        }
