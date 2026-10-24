from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import floor

import pandas as pd

# Pending buys expire after one 1-minute bar (fill-or-cancel on the next window).
DEFAULT_BUY_EXPIRE_MINUTES = 1


def integer_shares_for_notional(notional: float, price: float) -> int:
    """Largest whole-share qty with cost ``qty * price <= notional``."""
    px = float(price)
    notion = float(notional)
    if px <= 0 or notion <= 0:
        return 0
    return max(0, int(floor(notion / px)))


def buy_limit_fill_price(bar: pd.Series, limit_price: float) -> float | None:
    """Fill price for a buy limit on ``bar``, or None if not executable."""
    limit = float(limit_price)
    low = float(bar["low"])
    open_ = float(bar["open"])
    if low > limit:
        return None
    if open_ <= limit:
        return open_
    return limit


@dataclass
class OpenPosition:
    ticker: str
    qty: int
    entry_price: float
    entry_time: datetime
    limit_price: float
    stop_price: float
    trail_activate_price: float | None = None
    sell_trail: float | None = None
    trail_active: bool = False
    trail_stop_price: float | None = None
    position_id: int | None = None
    buy_order_id: int | None = None


@dataclass
class ClosedTrade:
    ticker: str
    qty: int
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    exit_reason: str
    realized_pnl: float
    position_id: int | None = None


@dataclass
class PendingBuy:
    """Submitted buy awaiting the next 1m window; cash already reserved."""

    ticker: str
    qty: int
    buy_limit_price: float
    reserved_cash: float
    submitted_at: datetime
    expire_at: datetime
    sell_limit_price: float
    stop_price: float
    trail_activate_price: float | None = None
    sell_trail: float | None = None
    order_id: int | None = None
    watch_id: int | None = None


@dataclass
class WatchlistEntry:
    """Transaction watchlist row for training/sim audit."""

    watch_id: int
    ticker: str
    side: str
    status: str  # pending | filled | canceled
    qty: int
    reserved_cash: float
    buy_limit_price: float
    submitted_at: datetime
    expire_at: datetime
    filled_at: datetime | None = None
    fill_price: float | None = None
    canceled_at: datetime | None = None
    cancel_reason: str | None = None
    order_id: int | None = None
    position_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "watch_id": self.watch_id,
            "ticker": self.ticker,
            "side": self.side,
            "status": self.status,
            "qty": self.qty,
            "reserved_cash": round(self.reserved_cash, 4),
            "buy_limit_price": self.buy_limit_price,
            "submitted_at": self.submitted_at.isoformat(),
            "expire_at": self.expire_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fill_price": self.fill_price,
            "canceled_at": self.canceled_at.isoformat() if self.canceled_at else None,
            "cancel_reason": self.cancel_reason,
            "order_id": self.order_id,
            "position_id": self.position_id,
        }


@dataclass
class PendingBuyResult:
    """Outcome of processing pending buys on one bar."""

    filled: list[OpenPosition] = field(default_factory=list)
    canceled: list[WatchlistEntry] = field(default_factory=list)


@dataclass
class SimBroker:
    """Cash + positions with pending buys, reservation, and limit/stop exits.

    Buys are submitted as **pending**: cash is reserved immediately and the
    order may fill on the next 1m bar only. If not filled by then, it cancels
    and reserved cash returns to the budget.

    Same-bar exit conflict: if both limit and stop would fill, **stop wins**.
    """

    cash: float
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    pending_buys: dict[str, PendingBuy] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    watchlist: list[WatchlistEntry] = field(default_factory=list)
    realized_pnl: float = 0.0
    _watch_seq: int = 0

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def pending_count(self) -> int:
        return len(self.pending_buys)

    @property
    def reserved_cash(self) -> float:
        return sum(p.reserved_cash for p in self.pending_buys.values())

    @property
    def available_cash(self) -> float:
        """Spendable cash (reservations already deducted from ``cash``)."""
        return self.cash

    @property
    def budget(self) -> float:
        """Cash on hand including amounts reserved for pending buys."""
        return self.cash + self.reserved_cash

    def has_position(self, ticker: str) -> bool:
        return ticker.upper() in self.positions

    def has_pending(self, ticker: str) -> bool:
        return ticker.upper() in self.pending_buys

    def can_open(self, *, max_positions: int, notional: float) -> bool:
        slots = self.open_count + self.pending_count
        if slots >= max(1, int(max_positions)):
            return False
        return self.cash >= float(notional)

    def submit_buy(
        self,
        ticker: str,
        *,
        buy_limit_price: float,
        notional: float,
        when: datetime,
        sell_limit_price: float,
        stop_price: float,
        trail_activate_price: float | None = None,
        sell_trail: float | None = None,
        expire_minutes: int = DEFAULT_BUY_EXPIRE_MINUTES,
        order_id: int | None = None,
    ) -> PendingBuy | None:
        """Place a pending buy: reserve cash; fill-or-cancel on next 1m window."""
        symbol = ticker.upper()
        if symbol in self.positions or symbol in self.pending_buys:
            return None
        limit = float(buy_limit_price)
        notion = float(notional)
        qty = integer_shares_for_notional(notion, limit)
        if qty < 1:
            return None
        reserved = qty * limit
        if reserved > notion or self.cash < reserved:
            return None

        self.cash -= reserved
        expire_at = when + timedelta(minutes=max(1, int(expire_minutes)))
        self._watch_seq += 1
        watch = WatchlistEntry(
            watch_id=self._watch_seq,
            ticker=symbol,
            side="buy",
            status="pending",
            qty=qty,
            reserved_cash=reserved,
            buy_limit_price=limit,
            submitted_at=when,
            expire_at=expire_at,
            order_id=order_id,
        )
        self.watchlist.append(watch)
        pending = PendingBuy(
            ticker=symbol,
            qty=qty,
            buy_limit_price=limit,
            reserved_cash=reserved,
            submitted_at=when,
            expire_at=expire_at,
            sell_limit_price=float(sell_limit_price),
            stop_price=float(stop_price),
            trail_activate_price=(
                float(trail_activate_price) if trail_activate_price is not None else None
            ),
            sell_trail=float(sell_trail) if sell_trail is not None else None,
            order_id=order_id,
            watch_id=watch.watch_id,
        )
        self.pending_buys[symbol] = pending
        return pending

    def process_pending_buys(
        self,
        bars_by_ticker: dict[str, pd.Series],
        when: datetime,
    ) -> PendingBuyResult:
        """Try to fill pending buys on this bar; cancel those past expire."""
        result = PendingBuyResult()
        for ticker in list(self.pending_buys.keys()):
            pending = self.pending_buys[ticker]
            # Same bar as submit: wait for the next 1m window.
            if when <= pending.submitted_at:
                continue
            bar = bars_by_ticker.get(ticker)
            fill_px = buy_limit_fill_price(bar, pending.buy_limit_price) if bar is not None else None

            if fill_px is not None:
                pos = self._fill_pending(pending, price=fill_px, when=when)
                if pos is not None:
                    result.filled.append(pos)
                continue

            if when >= pending.expire_at:
                canceled = self._cancel_pending(
                    pending, when=when, reason="expired_unfilled"
                )
                if canceled is not None:
                    result.canceled.append(canceled)
        return result

    def cancel_all_pending(
        self,
        when: datetime,
        *,
        reason: str = "session_end",
    ) -> list[WatchlistEntry]:
        canceled: list[WatchlistEntry] = []
        for ticker in list(self.pending_buys.keys()):
            entry = self._cancel_pending(self.pending_buys[ticker], when=when, reason=reason)
            if entry is not None:
                canceled.append(entry)
        return canceled

    def _watch_for(self, pending: PendingBuy) -> WatchlistEntry | None:
        if pending.watch_id is None:
            return None
        for entry in self.watchlist:
            if entry.watch_id == pending.watch_id:
                return entry
        return None

    def _fill_pending(
        self,
        pending: PendingBuy,
        *,
        price: float,
        when: datetime,
    ) -> OpenPosition | None:
        symbol = pending.ticker
        px = float(price)
        cost = pending.qty * px
        if cost > pending.reserved_cash + 1e-9:
            # Should not happen for limit fills at/below buy limit.
            return None
        refund = pending.reserved_cash - cost
        if refund > 0:
            self.cash += refund
        self.pending_buys.pop(symbol, None)
        pos = OpenPosition(
            ticker=symbol,
            qty=pending.qty,
            entry_price=px,
            entry_time=when,
            limit_price=pending.sell_limit_price,
            stop_price=pending.stop_price,
            trail_activate_price=pending.trail_activate_price,
            sell_trail=pending.sell_trail,
            buy_order_id=pending.order_id,
        )
        self.positions[symbol] = pos
        watch = self._watch_for(pending)
        if watch is not None:
            watch.status = "filled"
            watch.filled_at = when
            watch.fill_price = px
            watch.reserved_cash = cost
        return pos

    def _cancel_pending(
        self,
        pending: PendingBuy,
        *,
        when: datetime,
        reason: str,
    ) -> WatchlistEntry | None:
        symbol = pending.ticker
        if symbol not in self.pending_buys:
            return None
        self.cash += pending.reserved_cash
        self.pending_buys.pop(symbol, None)
        watch = self._watch_for(pending)
        if watch is not None:
            watch.status = "canceled"
            watch.canceled_at = when
            watch.cancel_reason = reason
        return watch

    def market_buy(
        self,
        ticker: str,
        *,
        price: float,
        notional: float,
        when: datetime,
        limit_price: float,
        stop_price: float,
        trail_activate_price: float | None = None,
        sell_trail: float | None = None,
        position_id: int | None = None,
        buy_order_id: int | None = None,
    ) -> OpenPosition | None:
        """Immediate whole-share buy (tests / forced fills). Prefer ``submit_buy`` in sim."""
        symbol = ticker.upper()
        if symbol in self.positions or symbol in self.pending_buys:
            return None
        px = float(price)
        notion = float(notional)
        qty = integer_shares_for_notional(notion, px)
        if qty < 1:
            return None
        cost = qty * px
        if cost > notion or self.cash < cost:
            return None
        self.cash -= cost
        pos = OpenPosition(
            ticker=symbol,
            qty=qty,
            entry_price=px,
            entry_time=when,
            limit_price=float(limit_price),
            stop_price=float(stop_price),
            trail_activate_price=(
                float(trail_activate_price) if trail_activate_price is not None else None
            ),
            sell_trail=float(sell_trail) if sell_trail is not None else None,
            position_id=position_id,
            buy_order_id=buy_order_id,
        )
        self.positions[symbol] = pos
        return pos

    def _update_trail(self, pos: OpenPosition, bar: pd.Series) -> None:
        if pos.trail_activate_price is None or pos.sell_trail is None:
            return
        high = float(bar["high"])
        if not pos.trail_active and high >= pos.trail_activate_price:
            pos.trail_active = True
            pos.trail_stop_price = pos.trail_activate_price * (1.0 - pos.sell_trail)
        if pos.trail_active:
            candidate = high * (1.0 - pos.sell_trail)
            if pos.trail_stop_price is None:
                pos.trail_stop_price = candidate
            else:
                pos.trail_stop_price = max(pos.trail_stop_price, candidate)

    def process_bar(self, ticker: str, bar: pd.Series, when: datetime) -> ClosedTrade | None:
        """Check exits for one ticker on the current bar. Stop beats limit same bar."""
        symbol = ticker.upper()
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        high = float(bar["high"])
        low = float(bar["low"])
        self._update_trail(pos, bar)

        stop_hit = low <= pos.stop_price
        trail_hit = (
            pos.trail_active
            and pos.trail_stop_price is not None
            and low <= pos.trail_stop_price
        )
        limit_hit = high >= pos.limit_price

        if stop_hit:
            return self._close(pos, price=pos.stop_price, when=when, reason="stop_loss")
        if trail_hit:
            return self._close(
                pos,
                price=float(pos.trail_stop_price),
                when=when,
                reason="trail_stop",
            )
        if limit_hit:
            return self._close(pos, price=pos.limit_price, when=when, reason="limit_profit")
        return None

    def process_all_open(self, bars_by_ticker: dict[str, pd.Series], when: datetime) -> list[ClosedTrade]:
        closed: list[ClosedTrade] = []
        for ticker in list(self.positions.keys()):
            bar = bars_by_ticker.get(ticker.upper())
            if bar is None:
                continue
            trade = self.process_bar(ticker, bar, when)
            if trade is not None:
                closed.append(trade)
        return closed

    def force_close_all(
        self,
        prices: dict[str, float],
        when: datetime,
        *,
        reason: str = "session_end",
    ) -> list[ClosedTrade]:
        closed: list[ClosedTrade] = []
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            px = float(prices.get(ticker, pos.entry_price))
            trade = self._close(pos, price=px, when=when, reason=reason)
            if trade is not None:
                closed.append(trade)
        return closed

    def _close(
        self,
        pos: OpenPosition,
        *,
        price: float,
        when: datetime,
        reason: str,
    ) -> ClosedTrade:
        proceeds = pos.qty * float(price)
        cost = pos.qty * pos.entry_price
        pnl = proceeds - cost
        self.cash += proceeds
        self.realized_pnl += pnl
        trade = ClosedTrade(
            ticker=pos.ticker,
            qty=pos.qty,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time,
            exit_price=float(price),
            exit_time=when,
            exit_reason=reason,
            realized_pnl=pnl,
            position_id=pos.position_id,
        )
        self.closed_trades.append(trade)
        self.positions.pop(pos.ticker, None)
        return trade
