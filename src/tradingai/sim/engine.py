from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from tradingai.intraday.bars import filter_rth_session
from tradingai.intraday.config import ET
from tradingai.sim.broker import (
    ClosedTrade,
    OpenPosition,
    SimBroker,
    WatchlistEntry,
    integer_shares_for_notional,
)
from tradingai.sim.clock import (
    as_of_bars,
    entry_cutoff_time,
    entry_window_start,
    session_days_in_sim_db,
)
from tradingai.sim.charts import DEFAULT_CHARTS_DIR, export_trade_charts
from tradingai.sim.config import (
    DEFAULT_QUERY_LATENCY_SECONDS,
    DEFAULT_QUERY_PACE_SECONDS,
    DEFAULT_SCAN_BATCH_SIZE,
    DEFAULT_START_CASH,
    get_sim_database_url,
)
from tradingai.sim.repository import (
    close_sim_position,
    create_sim_run,
    finalize_sim_run,
    initialize_sim_schema,
    insert_sim_fill,
    insert_sim_order,
    insert_sim_position,
    list_sim_tickers,
    load_prior_close,
    load_sim_bars_5m,
    update_sim_order,
)
from tradingai.sim.strategy import (
    DipObservation,
    EntrySignal,
    OpenDipStrategy,
    StrategyParams,
    entry_rank_key,
)


def _bar_series(bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if bars is None or bars.empty or ts not in bars.index:
        return None
    return bars.loc[ts]


def _as_datetime(ts: datetime | pd.Timestamp) -> datetime:
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime()
    return ts  # type: ignore[return-value]


class DipSimEngine:
    """Walk session days chronologically; step 5m bars with as-of visibility."""

    def __init__(
        self,
        *,
        params: StrategyParams | None = None,
        start_cash: float = DEFAULT_START_CASH,
        database_url: str | None = None,
        tickers: list[str] | None = None,
        persist: bool = True,
        query_latency_seconds: float = DEFAULT_QUERY_LATENCY_SECONDS,
        query_pace_seconds: float = DEFAULT_QUERY_PACE_SECONDS,
        scan_batch_size: int = DEFAULT_SCAN_BATCH_SIZE,
        scan_seed: int | None = None,
        export_charts: bool = True,
        charts_dir: str | None = None,
    ) -> None:
        self.params = params or StrategyParams()
        self.start_cash = float(start_cash)
        self.database_url = database_url or get_sim_database_url()
        self.tickers = [t.upper() for t in tickers] if tickers else None
        self.persist = persist
        self.query_latency_seconds = max(0.0, float(query_latency_seconds))
        self.query_pace_seconds = max(0.0, float(query_pace_seconds))
        self.scan_batch_size = max(1, int(scan_batch_size))
        # Unseeded → different permutation each engine / process; seed for tests.
        self.scan_seed = scan_seed
        self.export_charts = bool(export_charts)
        self.charts_dir = charts_dir or str(DEFAULT_CHARTS_DIR)
        self._scan_rng = random.Random(scan_seed)
        self.strategy = OpenDipStrategy(self.params)
        self.broker = SimBroker(cash=self.start_cash)
        self.run_id: int | None = None
        self.entries: list[EntrySignal] = []
        self.session_reports: list[dict[str, Any]] = []
        self._entered_today: set[str] = set()
        # Dip-qualified plungers carried to the next 5m bar for re-check.
        self._plunge_watch: set[str] = set()

    def run(
        self,
        *,
        start_day: date | None = None,
        end_day: date | None = None,
    ) -> dict[str, Any]:
        if self.persist:
            initialize_sim_schema(self.database_url)
            self.run_id = create_sim_run(
                params=self.params.to_dict(),
                start_cash=self.start_cash,
                database_url=self.database_url,
            )

        universe = self.tickers or list_sim_tickers(database_url=self.database_url)
        sessions = session_days_in_sim_db(database_url=self.database_url)
        if start_day is not None:
            sessions = [d for d in sessions if d >= start_day]
        if end_day is not None:
            sessions = [d for d in sessions if d <= end_day]

        # Preload all session bars once per day for the universe.
        session_total = len(sessions)
        for session_i, day in enumerate(sessions, start=1):
            self._run_session(
                day,
                universe,
                session_passed=session_i,
                session_total=session_total,
            )

        # Cancel leftover pendings and flatten open positions at sim end.
        if sessions:
            last_day = sessions[-1]
            last_ts: datetime = datetime(
                last_day.year, last_day.month, last_day.day, 15, 55, tzinfo=ET
            )
            for canceled in self.broker.cancel_all_pending(last_ts, reason="sim_end"):
                self._persist_buy_canceled(canceled)
            if self.broker.positions:
                prices: dict[str, float] = {}
                for ticker in list(self.broker.positions.keys()):
                    bars = load_sim_bars_5m(
                        ticker, session_day=last_day, database_url=self.database_url
                    )
                    if not bars.empty:
                        prices[ticker] = float(bars["close"].iloc[-1])
                        last_ts = bars.index[-1].to_pydatetime()
                closed = self.broker.force_close_all(prices, last_ts, reason="sim_end")
                for trade in closed:
                    self._persist_exit(trade)

        summary = self._build_summary(sessions, universe)
        if self.export_charts:
            chart_manifest = export_trade_charts(
                trades=self.broker.closed_trades,
                watchlist=self.broker.watchlist,
                database_url=self.database_url,
                out_dir=self.charts_dir,
                run_id=self.run_id,
            )
            summary["charts"] = chart_manifest
            print(
                f"charts: wrote {chart_manifest.get('charts_written', 0)} PNG(s) → "
                f"{chart_manifest.get('out_dir')}",
                flush=True,
            )
        if self.persist and self.run_id is not None:
            finalize_sim_run(
                self.run_id,
                end_cash=self.broker.cash,
                realized_pnl=self.broker.realized_pnl,
                trade_count=len(self.broker.closed_trades),
                win_count=sum(1 for t in self.broker.closed_trades if t.realized_pnl > 0),
                summary=summary,
                database_url=self.database_url,
            )
        return summary

    def _run_session(
        self,
        day: date,
        universe: list[str],
        *,
        session_passed: int = 1,
        session_total: int = 1,
    ) -> None:
        self._entered_today = set()
        self._plunge_watch = set()
        params = self.params
        entry_start = entry_window_start(day, window_minutes=params.window_minutes)
        entry_end = entry_cutoff_time(
            day,
            hour=params.entry_cutoff_hour,
            minute=params.entry_cutoff_minute,
        )
        realized_before = self.broker.realized_pnl
        trades_before = len(self.broker.closed_trades)

        bars_cache: dict[str, pd.DataFrame] = {}
        prior_cache: dict[str, float | None] = {}
        all_times: set[pd.Timestamp] = set()
        for ticker in universe:
            raw = load_sim_bars_5m(
                ticker, session_day=day, database_url=self.database_url
            )
            session = filter_rth_session(raw, session_day=day) if not raw.empty else raw
            if session.empty:
                continue
            bars_cache[ticker] = session
            prior_cache[ticker] = load_prior_close(
                ticker, day, database_url=self.database_url
            )
            all_times.update(pd.Timestamp(ts) for ts in session.index)

        if not all_times:
            self._print_session_report(
                day,
                session_pnl=0.0,
                trades=0,
                wins=0,
                entries=0,
                session_passed=session_passed,
                session_total=session_total,
            )
            return

        for ts in sorted(all_times):
            when = _as_datetime(ts)
            bar_map: dict[str, pd.Series] = {}
            for ticker, bars in bars_cache.items():
                series = _bar_series(bars, ts)
                if series is not None:
                    bar_map[ticker] = series

            # Pending buys: fill-or-cancel on the next 5m window after submit.
            pending_result = self.broker.process_pending_buys(bar_map, when)
            for pos in pending_result.filled:
                self._persist_buy_filled(pos, when)
            for canceled in pending_result.canceled:
                self._persist_buy_canceled(canceled)

            # Exits on open positions.
            for trade in self.broker.process_all_open(bar_map, when):
                self._persist_exit(trade)

            # New entries after dip window, before cutoff → submit as pending.
            if when < entry_start or when > entry_end:
                continue
            if not self.broker.can_open(
                max_positions=params.max_positions,
                notional=params.buy_notional_usd,
            ):
                continue

            self._scan_entries_at_bar(
                day=day,
                when=when,
                entry_end=entry_end,
                bars_cache=bars_cache,
                prior_cache=prior_cache,
            )

        # EOD: cancel unfilled pendings (release reserved cash) and flatten positions.
        last_ts = sorted(all_times)[-1]
        when = _as_datetime(last_ts)
        for canceled in self.broker.cancel_all_pending(when, reason="session_end"):
            self._persist_buy_canceled(canceled)
        if self.broker.positions:
            prices = {}
            for ticker in list(self.broker.positions.keys()):
                bars = bars_cache.get(ticker)
                if bars is not None and not bars.empty:
                    prices[ticker] = float(bars["close"].iloc[-1])
            for trade in self.broker.force_close_all(prices, when, reason="session_end"):
                self._persist_exit(trade)

        session_trades = self.broker.closed_trades[trades_before:]
        session_pnl = self.broker.realized_pnl - realized_before
        self._print_session_report(
            day,
            session_pnl=session_pnl,
            trades=len(session_trades),
            wins=sum(1 for t in session_trades if t.realized_pnl > 0),
            entries=len(self._entered_today),
            session_passed=session_passed,
            session_total=session_total,
        )

    def _print_session_report(
        self,
        day: date,
        *,
        session_pnl: float,
        trades: int,
        wins: int,
        entries: int,
        session_passed: int = 1,
        session_total: int = 1,
    ) -> None:
        balance = float(self.broker.cash)
        reserved = float(self.broker.reserved_cash)
        total_pnl = float(self.broker.realized_pnl)
        pending_n = self.broker.pending_count
        def _day_of(ts: datetime) -> date:
            if ts.tzinfo is None:
                return ts.date()
            return ts.astimezone(ET).date()

        watch = [
            w.to_dict()
            for w in self.broker.watchlist
            if _day_of(w.submitted_at) == day
            or (w.filled_at is not None and _day_of(w.filled_at) == day)
            or (w.canceled_at is not None and _day_of(w.canceled_at) == day)
        ]
        report = {
            "session_day": day.isoformat(),
            "session_passed": session_passed,
            "session_total": session_total,
            "session_pnl": round(session_pnl, 2),
            "balance": round(balance, 2),
            "reserved_cash": round(reserved, 2),
            "total_pnl": round(total_pnl, 2),
            "trades": trades,
            "wins": wins,
            "entries": entries,
            "pending_buys": pending_n,
            "watchlist": watch,
        }
        self.session_reports.append(report)
        print(
            f"{day.isoformat()}  "
            f"sessions={session_passed}/{session_total}  "
            f"session_pnl={session_pnl:+.2f}  "
            f"balance={balance:.2f}  "
            f"reserved={reserved:.2f}  "
            f"total_pnl={total_pnl:+.2f}  "
            f"trades={trades}  wins={wins}  entries={entries}  "
            f"pending={pending_n}",
            flush=True,
        )

    def _eligible_scan_tickers(self, bars_cache: dict[str, pd.DataFrame]) -> list[str]:
        return [
            t
            for t in bars_cache
            if t not in self._entered_today
            and not self.broker.has_position(t)
            and not self.broker.has_pending(t)
        ]

    def _scan_order(self, eligible: list[str]) -> list[str]:
        """Plunge-watch names first, then a fresh random walk of the rest."""
        watched = sorted(t for t in eligible if t in self._plunge_watch)
        rest = [t for t in eligible if t not in self._plunge_watch]
        self._scan_rng.shuffle(rest)
        return watched + rest

    def _observe_ticker(
        self,
        ticker: str,
        *,
        day: date,
        query_clock: datetime,
        bars_cache: dict[str, pd.DataFrame],
        prior_cache: dict[str, float | None],
    ) -> DipObservation:
        bars = bars_cache[ticker]
        visible = as_of_bars(bars, query_clock)
        obs = self.strategy.observe(
            ticker,
            visible,
            prior_close=prior_cache.get(ticker),
            session_day=day,
            as_of=query_clock,
        )
        if obs.is_plunging_dip:
            self._plunge_watch.add(ticker)
        else:
            self._plunge_watch.discard(ticker)
        return obs

    def _scan_entries_at_bar(
        self,
        *,
        day: date,
        when: datetime,
        entry_end: datetime,
        bars_cache: dict[str, pd.DataFrame],
        prior_cache: dict[str, float | None],
    ) -> None:
        """Batch-scan with IBKR-like virtual latency, then rank and enter.

        Walk order: plunge-watch (from prior 5m bar) first, then a random
        permutation of the remaining universe. After each batch of
        ``scan_batch_size`` queries, dip-analyze and enter in rank order
        (early reclaim 0<score<1, deeper dip first). Dip-qualified plungers
        stay on ``_plunge_watch`` for the next bar.
        """
        params = self.params
        latency = timedelta(seconds=self.query_latency_seconds)
        pace = timedelta(seconds=self.query_pace_seconds)
        batch_size = self.scan_batch_size
        query_clock = when
        order = self._scan_order(self._eligible_scan_tickers(bars_cache))
        query_index = 0
        offset = 0

        while offset < len(order):
            if query_clock > entry_end:
                break
            batch = order[offset : offset + batch_size]
            offset += len(batch)
            batch_obs: list[tuple[int, DipObservation]] = []

            for ticker in batch:
                query_clock = query_clock + latency
                if query_clock > entry_end:
                    # Unchecked remainder of this batch stays as previously watched.
                    break

                obs = self._observe_ticker(
                    ticker,
                    day=day,
                    query_clock=query_clock,
                    bars_cache=bars_cache,
                    prior_cache=prior_cache,
                )
                batch_obs.append((query_index, obs))
                query_index += 1

                query_clock = query_clock + pace
                if query_clock > entry_end:
                    break

            ranked = sorted(
                ((idx, obs) for idx, obs in batch_obs if obs.entry is not None),
                key=lambda item: entry_rank_key(item[1], query_index=item[0]),
            )
            for _, obs in ranked:
                if not self.broker.can_open(
                    max_positions=params.max_positions,
                    notional=params.buy_notional_usd,
                ):
                    break
                signal = obs.entry
                assert signal is not None
                if (
                    signal.ticker in self._entered_today
                    or self.broker.has_position(signal.ticker)
                    or self.broker.has_pending(signal.ticker)
                ):
                    continue
                if self._submit_buy(signal, signal.as_of):
                    self._entered_today.add(signal.ticker)

            if not self.broker.can_open(
                max_positions=params.max_positions,
                notional=params.buy_notional_usd,
            ):
                break
            if query_clock > entry_end:
                break

    def _submit_buy(self, signal: EntrySignal, when: datetime) -> bool:
        """Submit a pending buy (reserve cash). Fills on the next 5m bar or cancels."""
        qty = integer_shares_for_notional(
            self.params.buy_notional_usd, signal.buy_price
        )
        if qty < 1:
            return False

        buy_order_id = None
        if self.persist and self.run_id is not None:
            buy_order_id = insert_sim_order(
                self.run_id,
                ticker=signal.ticker,
                side="buy",
                order_type="limit",
                status="pending",
                qty=qty,
                created_at=when,
                limit_price=signal.buy_price,
                database_url=self.database_url,
            )

        pending = self.broker.submit_buy(
            signal.ticker,
            buy_limit_price=signal.buy_price,
            notional=self.params.buy_notional_usd,
            when=when,
            sell_limit_price=signal.limit_price,
            stop_price=signal.stop_price,
            trail_activate_price=signal.trail_activate_price,
            sell_trail=signal.sell_trail,
            order_id=buy_order_id,
        )
        if pending is None:
            return False
        self.entries.append(signal)
        self._plunge_watch.discard(signal.ticker)
        return True

    def _persist_buy_filled(self, pos: OpenPosition, when: datetime) -> None:
        if not self.persist or self.run_id is None:
            return
        position_id = insert_sim_position(
            self.run_id,
            ticker=pos.ticker,
            qty=pos.qty,
            entry_price=pos.entry_price,
            entry_time=when,
            limit_price=pos.limit_price,
            stop_price=pos.stop_price,
            trail_activate_price=pos.trail_activate_price,
            sell_trail=pos.sell_trail,
            database_url=self.database_url,
        )
        pos.position_id = position_id
        if pos.buy_order_id is not None:
            update_sim_order(
                pos.buy_order_id,
                status="filled",
                filled_at=when,
                fill_price=pos.entry_price,
                position_id=position_id,
                database_url=self.database_url,
            )
            insert_sim_fill(
                self.run_id,
                ticker=pos.ticker,
                side="buy",
                qty=pos.qty,
                price=pos.entry_price,
                filled_at=when,
                reason="entry",
                order_id=pos.buy_order_id,
                database_url=self.database_url,
            )
        insert_sim_order(
            self.run_id,
            ticker=pos.ticker,
            side="sell",
            order_type="limit",
            status="open",
            qty=pos.qty,
            created_at=when,
            limit_price=pos.limit_price,
            position_id=position_id,
            database_url=self.database_url,
        )
        insert_sim_order(
            self.run_id,
            ticker=pos.ticker,
            side="sell",
            order_type="stop",
            status="open",
            qty=pos.qty,
            created_at=when,
            stop_price=pos.stop_price,
            position_id=position_id,
            database_url=self.database_url,
        )
        for entry in self.broker.watchlist:
            if (
                entry.ticker == pos.ticker
                and entry.status == "filled"
                and entry.filled_at == when
            ):
                entry.position_id = position_id
                break

    def _persist_buy_canceled(self, entry: WatchlistEntry) -> None:
        if not self.persist or self.run_id is None:
            return
        if entry.order_id is None:
            return
        update_sim_order(
            entry.order_id,
            status="canceled",
            filled_at=entry.canceled_at,
            database_url=self.database_url,
        )

    def _persist_exit(self, trade: ClosedTrade) -> None:
        if not self.persist or self.run_id is None:
            return
        if trade.position_id is not None:
            close_sim_position(
                trade.position_id,
                exit_price=trade.exit_price,
                exit_time=trade.exit_time,
                exit_reason=trade.exit_reason,
                realized_pnl=trade.realized_pnl,
                database_url=self.database_url,
            )
        order_id = insert_sim_order(
            self.run_id,
            ticker=trade.ticker,
            side="sell",
            order_type=trade.exit_reason,
            status="filled",
            qty=trade.qty,
            created_at=trade.exit_time,
            filled_at=trade.exit_time,
            fill_price=trade.exit_price,
            position_id=trade.position_id,
            database_url=self.database_url,
        )
        insert_sim_fill(
            self.run_id,
            ticker=trade.ticker,
            side="sell",
            qty=trade.qty,
            price=trade.exit_price,
            filled_at=trade.exit_time,
            reason=trade.exit_reason,
            order_id=order_id,
            database_url=self.database_url,
        )

    def _build_summary(self, sessions: list[date], universe: list[str]) -> dict[str, Any]:
        trades = self.broker.closed_trades
        wins = [t for t in trades if t.realized_pnl > 0]
        losses = [t for t in trades if t.realized_pnl <= 0]
        by_reason: dict[str, int] = {}
        for t in trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        return {
            "run_id": self.run_id,
            "params": self.params.to_dict(),
            "query_latency_seconds": self.query_latency_seconds,
            "query_pace_seconds": self.query_pace_seconds,
            "scan_batch_size": self.scan_batch_size,
            "scan_seed": self.scan_seed,
            "sessions": [d.isoformat() for d in sessions],
            "session_reports": list(self.session_reports),
            "universe_size": len(universe),
            "start_cash": self.start_cash,
            "end_cash": self.broker.cash,
            "reserved_cash": self.broker.reserved_cash,
            "realized_pnl": self.broker.realized_pnl,
            "trade_count": len(trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": (len(wins) / len(trades)) if trades else 0.0,
            "exits_by_reason": by_reason,
            "entries": len(self.entries),
            "watchlist": [w.to_dict() for w in self.broker.watchlist],
            "watchlist_summary": {
                "pending": sum(1 for w in self.broker.watchlist if w.status == "pending"),
                "filled": sum(1 for w in self.broker.watchlist if w.status == "filled"),
                "canceled": sum(
                    1 for w in self.broker.watchlist if w.status == "canceled"
                ),
            },
            "trades": [
                {
                    "ticker": t.ticker,
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_price": t.exit_price,
                    "exit_time": t.exit_time.isoformat(),
                    "exit_reason": t.exit_reason,
                    "realized_pnl": t.realized_pnl,
                }
                for t in trades
            ],
            "disclaimer": (
                "Simulated open-dip trading over Alpaca 5m bars. "
                "Buys are pending with cash reserved until the next 5m bar "
                "(fill-or-cancel). Informational only — not financial advice."
            ),
        }
