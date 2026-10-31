"""IBKR TWS historical bars for tradingai-sync-intraday (ib_insync).

Not registered as the global TRADINGAI_DATA_PROVIDER — sync-only.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from tradingai.intraday.bars import normalize_intraday_index
from tradingai.intraday.config import (
    ET,
    get_bar_interval,
    get_bar_period,
    get_ibkr_client_id,
    get_ibkr_host,
    get_ibkr_port,
)

# Fraction of total_budget spent on a budget limit buy (1.0 = full budget).
DEFAULT_BUDGET_FRACTION = 1.0
# Attached SELL stop under the buy limit (fraction; 0.01 = 1%). None = no stop.
DEFAULT_BUY_STOP_LOSS: float | None = None
# IB ``trailingPercent`` units are percent points (0.05 == 0.05%).
DEFAULT_SELL_TRAIL_PERCENT = 0.05
# sell-trail: wait until fee-aware net return reaches this, then place trail.
DEFAULT_SELL_TRAIL_ACTIVATE_RETURN = 0.002  # 0.2%
# sell-trail stepped tighten (IB percent points) at 1.5× / 2× activation profit.
DEFAULT_SELL_TRAIL_1_5X_PERCENT = 0.03
DEFAULT_SELL_TRAIL_2X_PERCENT = 0.01
# Protective trail after budget buy (buy-trail); IB percent points.
DEFAULT_BUY_TRAIL_PERCENT = 0.6
# Tightened trail once fee-aware 1× target mid is reached (IB percent points).
DEFAULT_BUY_TRAIL_TIGHT_PERCENT = 0.05
# Further tighten at 1.5× / 2× fee-aware target return (IB percent points).
DEFAULT_BUY_TRAIL_1_5X_PERCENT = 0.03
DEFAULT_BUY_TRAIL_2X_PERCENT = 0.01

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


def format_ib_end_datetime(value: datetime | str | None) -> str:
    """Format an inclusive request end for IB ``endDateTime``.

    Naive datetimes are interpreted as America/New_York. Aware datetimes are
    converted to UTC, which is the form ``ib_insync`` sends for datetimes.
    ``None`` or ``""`` means "now".
    """
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=ET)
    utc = value.astimezone(timezone.utc)
    return utc.strftime("%Y%m%d %H:%M:%S UTC")


def qualify_equity(ib, ticker: str):
    """Qualify a SMART USD stock contract. Share classes use an IB space."""
    from ib_insync import Stock

    symbol = ticker.upper().strip()
    ib_symbol = symbol.replace(".", " ").replace("-", " ")
    contract = Stock(ib_symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise ValueError(f"IBKR could not qualify contract for {symbol}.")
    return qualified[0]


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
    end_datetime: datetime | str | None = None,
    contract=None,
) -> pd.DataFrame:
    """Fetch historical OHLCV for one equity via an open IB connection.

    ``end_datetime`` sets IB ``endDateTime``. Leave it empty to request up to
    now. Pass a pre-qualified ``contract`` to skip a qualify call per day.
    """
    duration = period_to_ib_duration(period)
    bar_size = interval_to_ib_bar_size(interval)
    qualified = contract if contract is not None else qualify_equity(ib, ticker)
    bars = ib.reqHistoricalData(
        qualified,
        endDateTime=format_ib_end_datetime(end_datetime),
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


def fetch_order_snapshot(
    order_id: int,
    *,
    symbol: str | None = None,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict:
    """Look up an order by id among open trades and recent fills.

    Returns fill progress plus whether the order is still working.
    """
    oid = int(order_id)
    ticker = symbol.upper().strip() if symbol else None
    with connect_ib(
        host=host,
        port=port,
        client_id=client_id,
        readonly=True,
    ) as ib:
        ib.reqOpenOrders()
        ib.sleep(0.35)
        open_match = None
        for trade in ib.openTrades() or []:
            order = trade.order
            if order is None or int(getattr(order, "orderId", -1)) != oid:
                continue
            contract = trade.contract
            sym = getattr(contract, "symbol", None)
            if ticker and sym and str(sym).upper() != ticker:
                continue
            status = trade.orderStatus
            open_match = {
                "order_id": oid,
                "symbol": sym,
                "action": getattr(order, "action", None),
                "order_type": getattr(order, "orderType", None),
                "total_quantity": float(getattr(order, "totalQuantity", 0) or 0),
                "status": status.status if status else None,
                "filled": float(status.filled) if status else 0.0,
                "remaining": float(status.remaining) if status else None,
                "avg_fill_price": (
                    float(status.avgFillPrice) if status and status.avgFillPrice else None
                ),
                "open": True,
            }
            break

        fills = []
        for fill in ib.fills() or []:
            execution = fill.execution
            if execution is None or int(getattr(execution, "orderId", -1)) != oid:
                continue
            fills.append(serialize_fill(fill))

    filled_shares = sum(float(f.get("shares") or 0.0) for f in fills)
    return {
        "order_id": oid,
        "symbol": ticker or (open_match or {}).get("symbol"),
        "open": open_match is not None,
        "open_order": open_match,
        "fills": fills,
        "filled_shares": filled_shares,
        "buy_complete": open_match is None and filled_shares > 0,
        "source": "ibkr_tws",
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


def _finite_positive(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        px = float(value)
    except (TypeError, ValueError):
        return None
    if px != px or px <= 0.0:  # NaN or non-positive
        return None
    return px


def round_us_equity_price(price: float, *, mode: str = "round") -> float:
    """Round a USD equity price to a valid IB tick.

    Stocks at/above $1 trade in $0.01 increments; below $1 in $0.0001.
    ``mode``: ``round`` (nearest), ``ceil`` (up), or ``floor`` (down).
    """
    px = float(price)
    if px != px or px <= 0.0:
        raise ValueError(f"price must be positive, got {price!r}")
    tick = 0.01 if px >= 1.0 else 0.0001
    steps = px / tick
    if mode == "ceil":
        stepped = math.ceil(steps - 1e-12)
    elif mode == "floor":
        stepped = math.floor(steps + 1e-12)
    elif mode == "round":
        stepped = round(steps)
    else:
        raise ValueError(f"Unknown rounding mode {mode!r}")
    # Avoid float dust (e.g. 843.4600000001).
    decimals = 2 if tick >= 0.01 else 4
    return round(stepped * tick, decimals)


def round_us_equity_offset(offset: float) -> float:
    """Round a dollar offset (e.g. TRAIL LIMIT lmtPriceOffset) to a cent tick."""
    off = float(offset)
    if off != off or off <= 0.0:
        raise ValueError(f"offset must be positive, got {offset!r}")
    # Offsets are always dollar increments for US stocks — never sub-penny.
    return max(0.01, round(off, 2))


def _quote_from_ticker(tick) -> tuple[float | None, str, dict]:
    """Pick mid → bid/ask mid → last → close from an ib_insync Ticker."""
    mid = _finite_positive(tick.midpoint() if hasattr(tick, "midpoint") else None)
    bid = _finite_positive(getattr(tick, "bid", None))
    ask = _finite_positive(getattr(tick, "ask", None))
    last = _finite_positive(getattr(tick, "last", None))
    close = _finite_positive(getattr(tick, "close", None))
    fields = {"bid": bid, "ask": ask, "last": last, "close": close}
    if mid is not None:
        return mid, "mid", fields
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0, "bid_ask_mid", fields
    if last is not None:
        return last, "last", fields
    if close is not None:
        return close, "close", fields
    return None, "none", fields


def fetch_equity_mid_price(
    symbol: str,
    *,
    wait_seconds: float = 1.5,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict:
    """Fetch current mid (or bid/ask midpoint / last) for a SMART/USD equity.

    Tries live market data first; on empty quotes (common IB Error 10089 when
    TOP/ALL is not subscribed for API), falls back to **delayed** data
    (``reqMarketDataType(3)``).
    """
    ticker = symbol.upper().strip()
    if not ticker:
        raise ValueError("symbol is required.")

    # 1=live, 3=delayed (IB: "Delayed market data is available" on Error 10089).
    md_attempts: tuple[tuple[int, str], ...] = ((1, "live"), (3, "delayed"))

    with connect_ib(
        host=host,
        port=port,
        client_id=client_id,
        readonly=True,
    ) as ib:
        contract = qualify_equity(ib, ticker)
        last_fields: dict = {"bid": None, "ask": None, "last": None, "close": None}
        for md_type, md_label in md_attempts:
            try:
                ib.reqMarketDataType(md_type)
            except Exception:
                pass
            tick = ib.reqMktData(contract, "", False, False)
            try:
                if wait_seconds > 0:
                    ib.sleep(wait_seconds)
                price, source, fields = _quote_from_ticker(tick)
                last_fields = fields
                if price is not None:
                    return {
                        "symbol": ticker,
                        "mid_price": round(float(price), 6),
                        "price_source": source,
                        "market_data_type": md_label,
                        "bid": fields["bid"],
                        "ask": fields["ask"],
                        "last": fields["last"],
                        "close": fields["close"],
                        "source": "ibkr_tws",
                    }
            finally:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass
        raise ValueError(
            f"No usable quote for {ticker} (mid/bid/ask/last/close empty; "
            f"tried live+delayed). Last fields={last_fields}. "
            "IB Error 10089 means live TOP is not subscribed for API — "
            "delayed should work if Market Data Farm is connected; "
            "or pass an explicit --price / limit."
        )


def integer_shares_for_budget(budget_usd: float, price: float) -> int:
    """Largest whole-share qty with ``qty * price <= budget_usd``."""
    notion = float(budget_usd)
    px = float(price)
    if notion <= 0.0 or px <= 0.0:
        return 0
    return max(0, int(math.floor(notion / px)))


def place_budget_limit_buy(
    *,
    symbol: str,
    total_budget: float,
    limit_price: float | None = None,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    stop_loss: float | None = DEFAULT_BUY_STOP_LOSS,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    dry_run: bool = True,
    allow_orders: bool = False,
    max_shares: float = 1_000.0,
    wait_seconds: float = 2.0,
    quote_wait_seconds: float = 1.5,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict:
    """Buy integer shares with a LIMIT order sized to a budget fraction.

    Uses ``floor((total_budget * budget_fraction) / limit_price)`` shares.
    When ``limit_price`` is omitted, fetches the current mid from TWS.
    When ``stop_loss`` is set (e.g. ``0.01`` = 1%), attaches a child SELL STP
    at ``limit_price * (1 - stop_loss)``. Default is no stop
    (``stop_loss=None`` or ``0``).
    """
    ticker = symbol.upper().strip()
    if not ticker:
        raise ValueError("symbol is required.")
    budget = float(total_budget)
    if budget <= 0.0:
        raise ValueError("total_budget must be positive.")
    frac = float(budget_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("budget_fraction must be in (0, 1].")
    spend = budget * frac

    sl: float | None
    if stop_loss is None or float(stop_loss) <= 0.0:
        sl = None
    else:
        sl = float(stop_loss)
        if sl >= 1.0:
            raise ValueError("stop_loss must be a fraction in (0, 1), e.g. 0.01 for 1%.")

    quote: dict | None = None
    price = _finite_positive(limit_price)
    if price is None:
        quote = fetch_equity_mid_price(
            ticker,
            wait_seconds=quote_wait_seconds,
            host=host,
            port=port,
            client_id=client_id,
        )
        price = float(quote["mid_price"])

    # Buy limit / stop must be on valid IB ticks (Error 201 otherwise).
    price = round_us_equity_price(float(price), mode="round")
    qty = integer_shares_for_budget(spend, price)
    stop_price = (
        round_us_equity_price(float(price) * (1.0 - sl), mode="floor")
        if sl is not None
        else None
    )
    sizing = {
        "symbol": ticker,
        "total_budget": round(budget, 2),
        "budget_fraction": frac,
        "spend_budget": round(spend, 2),
        "limit_price": price,
        "quantity": qty,
        "notional": round(qty * float(price), 2),
        "stop_loss": sl,
        "stop_price": stop_price,
        "order_type": "LMT",
        "side": "buy",
        "quote": quote,
    }
    if qty < 1:
        return {
            **sizing,
            "submitted": False,
            "status": "rejected",
            "error": (
                f"Budget too small for 1 share at {price}: "
                f"spend_budget={spend:.2f}."
            ),
            "dry_run": bool(dry_run),
            "source": "ibkr_tws",
            "disclaimer": (
                "IBKR order helper. Real money risk if not on paper. "
                "Not financial advice."
            ),
        }

    # No stop → existing single-order path.
    if sl is None:
        order = place_equity_order(
            symbol=ticker,
            side="buy",
            quantity=float(qty),
            order_type="LMT",
            limit_price=float(price),
            tif=tif,
            outside_rth=outside_rth,
            account=account,
            dry_run=dry_run,
            allow_orders=allow_orders,
            max_shares=max_shares,
            wait_seconds=wait_seconds,
            host=host,
            port=port,
            client_id=client_id,
        )
        return {
            **order,
            "stop_loss": None,
            "stop_price": None,
            "sizing": sizing,
            "quote": quote,
        }

    tif_norm = (tif or "DAY").strip().upper()
    preview = {
        "symbol": ticker,
        "side": "buy",
        "quantity": float(qty),
        "order_type": "LMT",
        "limit_price": price,
        "stop_loss": sl,
        "stop_price": stop_price,
        "tif": tif_norm,
        "outside_rth": bool(outside_rth),
        "account": account,
        "exchange": "SMART",
        "currency": "USD",
        "dry_run": bool(dry_run),
        "sizing": sizing,
        "quote": quote,
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
            "Dry run only — parent LMT + child STP not sent. "
            "Call with dry_run=false and TRADINGAI_IBKR_ALLOW_ORDERS=1 to place."
        )
        return preview

    if not allow_orders:
        raise PermissionError(
            "Order placement blocked. Set TRADINGAI_IBKR_ALLOW_ORDERS=1 "
            "and pass dry_run=false to submit."
        )
    if max_shares > 0.0 and qty > max_shares:
        raise ValueError(
            f"quantity {qty} exceeds max_shares={max_shares}. "
            "Raise TRADINGAI_IBKR_MAX_ORDER_SHARES if intentional."
        )

    from ib_insync import LimitOrder, Stock, StopOrder

    ib_symbol = ticker.replace(".", " ").replace("-", " ")
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

        parent = LimitOrder("BUY", qty, float(price))
        parent.orderId = ib.client.getReqId()
        parent.tif = tif_norm
        parent.outsideRth = bool(outside_rth)
        parent.transmit = False
        if account:
            parent.account = account.strip()

        assert stop_price is not None
        child = StopOrder("SELL", qty, float(stop_price))
        child.parentId = parent.orderId
        child.tif = tif_norm
        child.outsideRth = bool(outside_rth)
        child.transmit = True
        if account:
            child.account = account.strip()

        parent_trade = ib.placeOrder(contract, parent)
        child_trade = ib.placeOrder(contract, child)
        if wait_seconds > 0:
            ib.sleep(wait_seconds)

        order_status = parent_trade.orderStatus
        fills = [serialize_fill(fill) for fill in (parent_trade.fills or [])]
        return {
            **preview,
            "status": order_status.status if order_status else "submitted",
            "submitted": True,
            "order_id": parent.orderId,
            "stop_order_id": getattr(child_trade.order, "orderId", None),
            "perm_id": parent_trade.order.permId if parent_trade.order else None,
            "filled": float(order_status.filled) if order_status else 0.0,
            "remaining": float(order_status.remaining) if order_status else float(qty),
            "avg_fill_price": (
                float(order_status.avgFillPrice) if order_status else None
            ),
            "why_held": getattr(order_status, "whyHeld", None) or None,
            "log": [str(entry) for entry in (parent_trade.log or [])][-8:],
            "fills": fills,
            "message": (
                f"Bracket submitted to TWS "
                f"(parent={order_status.status if order_status else 'unknown'}, "
                f"stop@{stop_price})."
            ),
        }


def fetch_equity_position(
    symbol: str,
    *,
    account: str | None = None,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict | None:
    """Return the open SMART/USD equity position for ``symbol``, or None."""
    ticker = symbol.upper().strip()
    if not ticker:
        raise ValueError("symbol is required.")
    account_filter = account.strip() if account else None

    with connect_ib(
        host=host,
        port=port,
        client_id=client_id,
        readonly=True,
    ) as ib:
        positions = list(ib.reqPositions() or [])
        matches = []
        for pos in positions:
            contract = pos.contract
            sym = (getattr(contract, "symbol", None) or "").upper()
            local = (getattr(contract, "localSymbol", None) or "").upper()
            sec = (getattr(contract, "secType", None) or "").upper()
            if sec and sec != "STK":
                continue
            if sym != ticker and local.replace(" ", ".") != ticker and local != ticker:
                continue
            acct = getattr(pos, "account", None) or None
            if account_filter and acct != account_filter:
                continue
            qty = float(pos.position)
            if abs(qty) < 1e-12:
                continue
            matches.append(
                {
                    "symbol": ticker,
                    "account": acct,
                    "quantity": qty,
                    "avg_cost": float(pos.avgCost) if pos.avgCost is not None else None,
                    "sec_type": sec or "STK",
                    "source": "ibkr_tws",
                }
            )
        if not matches:
            return None
        # Prefer long inventory; otherwise first match.
        longs = [m for m in matches if m["quantity"] > 0]
        return longs[0] if longs else matches[0]


def fetch_last_buy_price(
    symbol: str,
    *,
    account: str | None = None,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict | None:
    """Most recent buy fill price for ``symbol`` (session executions)."""
    payload = fetch_recent_trades(
        limit=100,
        side="buy",
        symbol=symbol,
        account=account,
        host=host,
        port=port,
        client_id=client_id,
    )
    trades = payload.get("trades") or []
    if not trades:
        return None
    # fetch_recent_trades already newest-first
    top = trades[0]
    price = _finite_positive(top.get("price") or top.get("avg_price"))
    if price is None:
        return None
    return {
        "symbol": symbol.upper().strip(),
        "buy_price": price,
        "shares": float(top.get("shares") or 0.0),
        "time": top.get("time"),
        "account": top.get("account"),
        "source": "ibkr_tws_execution",
    }


def resolve_sell_limit_sizing(
    *,
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    target_return: float | None = None,
    account: str | None = None,
    dry_run: bool = True,
    max_shares: float = 1_000.0,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    order_type: str = "LMT",
) -> dict:
    """Resolve qty + limit for a sell (full position / fee-aware target defaults).

    Returns a sizing dict. On missing long position when qty is omitted,
    returns ``status=rejected`` with an ``error`` (no exception).
    """
    from tradingai.intraday.config import get_target_return
    from tradingai.intraday.profit_target import fee_aware_sell_price

    ticker = symbol.upper().strip()
    if not ticker:
        raise ValueError("symbol is required.")

    position = None
    buy_info = None
    buy_price = None

    if quantity is None or limit_price is None:
        try:
            position = fetch_equity_position(
                ticker,
                account=account,
                host=host,
                port=port,
                client_id=client_id,
            )
        except Exception as exc:
            if quantity is None:
                raise ValueError(
                    f"Could not load open position for {ticker}: {exc}"
                ) from exc
        try:
            buy_info = fetch_last_buy_price(
                ticker,
                account=account,
                host=host,
                port=port,
                client_id=client_id,
            )
        except Exception:
            buy_info = None

    if quantity is None:
        if position is None or float(position["quantity"]) <= 0:
            return {
                "symbol": ticker,
                "side": "sell",
                "order_type": order_type,
                "submitted": False,
                "status": "rejected",
                "error": f"No open long position found for {ticker}.",
                "dry_run": bool(dry_run),
                "source": "ibkr_tws",
                "disclaimer": (
                    "IBKR order helper. Real money risk if not on paper. "
                    "Not financial advice."
                ),
            }
        qty = float(math.floor(float(position["quantity"])))
    else:
        qty = float(quantity)
    if qty <= 0:
        raise ValueError("quantity must be positive.")
    if max_shares > 0.0 and qty > max_shares:
        raise ValueError(
            f"quantity {qty} exceeds max_shares={max_shares}. "
            "Raise TRADINGAI_IBKR_MAX_ORDER_SHARES if intentional."
        )

    if buy_info and buy_info.get("buy_price"):
        buy_price = float(buy_info["buy_price"])
        buy_source = "last_buy_fill"
    elif position and position.get("avg_cost"):
        buy_price = float(position["avg_cost"])
        buy_source = "position_avg_cost"
    else:
        buy_source = None

    ret = (
        float(target_return)
        if target_return is not None
        else float(get_target_return())
    )
    target_meta = None
    price = _finite_positive(limit_price)
    if price is None:
        if buy_price is None or buy_price <= 0:
            raise ValueError(
                "limit_price omitted and no last buy / avg cost available "
                f"for {ticker}. Pass --price explicitly."
            )
        notional = buy_price * qty
        target_meta = fee_aware_sell_price(
            buy_price,
            buy_notional_usd=notional,
            net_return=ret,
        )
        price = round_us_equity_price(
            float(target_meta["sell_price"]), mode="ceil"
        )
    else:
        price = round_us_equity_price(float(price), mode="round")

    return {
        "symbol": ticker,
        "quantity": qty,
        "limit_price": price,
        "buy_price": round(float(buy_price), 6) if buy_price else None,
        "buy_price_source": buy_source,
        "target_return": ret,
        "target_meta": target_meta,
        "position": position,
        "account": account or (position.get("account") if position else None),
        "order_type": order_type,
        "side": "sell",
        "dry_run": bool(dry_run),
        "source": "ibkr_tws",
        "disclaimer": (
            "IBKR order helper. Real money risk if not on paper. "
            "Not financial advice."
        ),
    }


def place_limit_sell(
    *,
    symbol: str,
    quantity: float | None = None,
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
    target_return: float | None = None,
) -> dict:
    """Sell with a plain IBKR LIMIT order.

    Defaults:
    - quantity = full long position for the symbol
    - limit_price = fee-aware sell target at ``target_return`` on last buy /
      position avg cost (``TRADINGAI_OPEN_DIP_TARGET_RETURN`` / DEFAULT)
    """
    sizing = resolve_sell_limit_sizing(
        symbol=symbol,
        quantity=quantity,
        limit_price=limit_price,
        target_return=target_return,
        account=account,
        dry_run=dry_run,
        max_shares=max_shares,
        host=host,
        port=port,
        client_id=client_id,
        order_type="LMT",
    )
    if sizing.get("status") == "rejected":
        return sizing

    qty = float(sizing["quantity"])
    price = float(sizing["limit_price"])
    preview = {
        "symbol": sizing["symbol"],
        "side": "sell",
        "quantity": qty,
        "order_type": "LMT",
        "limit_price": price,
        "tif": (tif or "DAY").strip().upper(),
        "outside_rth": bool(outside_rth),
        "account": sizing.get("account"),
        "exchange": "SMART",
        "currency": "USD",
        "dry_run": bool(dry_run),
        "sizing": sizing,
        "source": "ibkr_tws",
        "disclaimer": sizing["disclaimer"],
        "target_return": sizing.get("target_return"),
    }

    order = place_equity_order(
        symbol=sizing["symbol"],
        side="sell",
        quantity=qty,
        order_type="LMT",
        limit_price=price,
        tif=tif,
        outside_rth=outside_rth,
        account=sizing.get("account"),
        dry_run=dry_run,
        allow_orders=allow_orders,
        max_shares=max_shares,
        wait_seconds=wait_seconds,
        host=host,
        port=port,
        client_id=client_id,
    )
    return {**preview, **order, "sizing": sizing, "order_type": "LMT"}


def place_trail_limit_sell(
    *,
    symbol: str,
    quantity: float | None = None,
    limit_price: float | None = None,
    trail_percent: float = DEFAULT_SELL_TRAIL_PERCENT,
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
    target_return: float | None = None,
) -> dict:
    """Sell with an IBKR TRAIL LIMIT order.

    Defaults:
    - quantity = full long position for the symbol
    - limit_price = fee-aware sell target at ``target_return`` on last buy /
      position avg cost (``TRADINGAI_OPEN_DIP_TARGET_RETURN`` / DEFAULT)
    - trail_percent = 0.05 (IB percent points → 0.05%)
    """
    from ib_insync import Order, Stock

    trail = float(trail_percent)
    if trail <= 0.0:
        raise ValueError("trail_percent must be positive.")

    sizing = resolve_sell_limit_sizing(
        symbol=symbol,
        quantity=quantity,
        limit_price=limit_price,
        target_return=target_return,
        account=account,
        dry_run=dry_run,
        max_shares=max_shares,
        host=host,
        port=port,
        client_id=client_id,
        order_type="TRAIL LIMIT",
    )
    if sizing.get("status") == "rejected":
        return sizing

    ticker = sizing["symbol"]
    qty = float(sizing["quantity"])
    price = float(sizing["limit_price"])
    position = sizing.get("position")
    lmt_offset = round_us_equity_offset(
        max(0.01, float(price) * (trail / 100.0))
    )
    sizing = {
        **sizing,
        "trail_percent": trail,
        "lmt_price_offset": lmt_offset,
        "trail_stop_price": price,
        "order_type": "TRAIL LIMIT",
    }

    preview = {
        "symbol": ticker,
        "side": "sell",
        "quantity": qty,
        "order_type": "TRAIL LIMIT",
        "limit_price": price,
        "trail_percent": trail,
        "lmt_price_offset": lmt_offset,
        "trail_stop_price": price,
        "tif": (tif or "DAY").strip().upper(),
        "outside_rth": bool(outside_rth),
        "account": sizing.get("account"),
        "exchange": "SMART",
        "currency": "USD",
        "dry_run": bool(dry_run),
        "sizing": sizing,
        "source": "ibkr_tws",
        "disclaimer": sizing["disclaimer"],
        "target_return": sizing.get("target_return"),
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

    ib_symbol = ticker.replace(".", " ").replace("-", " ")
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

        order = Order()
        order.action = "SELL"
        order.orderType = "TRAIL LIMIT"
        order.totalQuantity = qty
        order.trailingPercent = trail
        order.lmtPriceOffset = float(lmt_offset)
        order.trailStopPrice = float(price)
        order.tif = (tif or "DAY").strip().upper()
        order.outsideRth = bool(outside_rth)
        acct = sizing.get("account")
        if acct:
            order.account = str(acct).strip()

        trade = ib.placeOrder(contract, order)
        if wait_seconds > 0:
            ib.sleep(wait_seconds)

        order_status = trade.orderStatus
        status_name = order_status.status if order_status else "submitted"
        fills = [serialize_fill(fill) for fill in (trade.fills or [])]
        log_entries = [str(entry) for entry in (trade.log or [])][-8:]
        rejected = status_name in {
            "Inactive",
            "Cancelled",
            "ApiCancelled",
            "Rejected",
        } or any(
            "Error 201" in entry or "Invalid Price" in entry
            for entry in log_entries
        )
        if rejected:
            msg = (
                f"Order rejected by TWS ({status_name}). "
                "Check log for Invalid Price / Error 201."
            )
        else:
            msg = f"Order submitted to TWS ({status_name})."
        return {
            **preview,
            "status": status_name,
            "submitted": not rejected,
            "order_id": trade.order.orderId if trade.order else None,
            "perm_id": trade.order.permId if trade.order else None,
            "filled": float(order_status.filled) if order_status else 0.0,
            "remaining": float(order_status.remaining) if order_status else qty,
            "avg_fill_price": (
                float(order_status.avgFillPrice) if order_status else None
            ),
            "why_held": getattr(order_status, "whyHeld", None) or None,
            "log": log_entries,
            "fills": fills,
            "message": msg,
            "error": (
                "Order rejected — Invalid Price or cancelled by IB."
                if rejected
                else None
            ),
        }


def modify_trail_limit_percent(
    *,
    order_id: int,
    trail_percent: float,
    trail_stop_price: float | None = None,
    symbol: str | None = None,
    wait_seconds: float = 1.0,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
) -> dict:
    """Update ``trailingPercent`` (and optional ``trailStopPrice``) on a working TRAIL LIMIT."""
    oid = int(order_id)
    trail = float(trail_percent)
    if trail <= 0.0:
        raise ValueError("trail_percent must be positive.")
    ticker = symbol.upper().strip() if symbol else None

    with connect_ib(
        host=host,
        port=port,
        client_id=client_id,
        readonly=False,
    ) as ib:
        ib.reqOpenOrders()
        ib.sleep(0.35)
        match = None
        for trade in ib.openTrades() or []:
            order = trade.order
            if order is None or int(getattr(order, "orderId", -1)) != oid:
                continue
            contract = trade.contract
            sym = getattr(contract, "symbol", None)
            if ticker and sym and str(sym).upper() != ticker:
                continue
            match = trade
            break
        if match is None:
            return {
                "order_id": oid,
                "submitted": False,
                "status": "not_found",
                "error": f"No open order with order_id={oid}.",
                "source": "ibkr_tws",
            }

        order = match.order
        contract = match.contract
        order.trailingPercent = trail
        stop_px = _finite_positive(trail_stop_price)
        if stop_px is not None:
            stop_px = round_us_equity_price(float(stop_px), mode="floor")
            order.trailStopPrice = float(stop_px)
            order.lmtPriceOffset = float(
                round_us_equity_offset(max(0.01, float(stop_px) * (trail / 100.0)))
            )
        elif order.trailStopPrice:
            order.lmtPriceOffset = float(
                round_us_equity_offset(
                    max(0.01, float(order.trailStopPrice) * (trail / 100.0))
                )
            )

        trade = ib.placeOrder(contract, order)
        if wait_seconds > 0:
            ib.sleep(wait_seconds)
        status = trade.orderStatus
        return {
            "order_id": oid,
            "symbol": getattr(contract, "symbol", ticker),
            "trail_percent": trail,
            "trail_stop_price": float(order.trailStopPrice)
            if order.trailStopPrice
            else None,
            "lmt_price_offset": float(order.lmtPriceOffset)
            if order.lmtPriceOffset
            else None,
            "status": status.status if status else "submitted",
            "submitted": True,
            "source": "ibkr_tws",
            "message": f"TRAIL LIMIT {oid} updated to trail={trail}%.",
        }
