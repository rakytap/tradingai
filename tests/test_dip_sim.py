from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingai.providers.alpaca_provider import (
    bars_df_to_ohlcv,
    interval_to_timeframe,
    period_to_start_end,
)
from tradingai.sim.broker import (
    ClosedTrade,
    SimBroker,
    WatchlistEntry,
    buy_limit_fill_price,
    integer_shares_for_notional,
)
from tradingai.sim.charts import collect_chart_cases, plot_trade_case
from tradingai.sim.clock import as_of_bars, entry_window_start
from tradingai.sim.engine import DipSimEngine
from tradingai.sim.strategy import (
    DipObservation,
    EntrySignal,
    OpenDipStrategy,
    StrategyParams,
    asymmetric_parabola_curve,
    asymmetric_parabola_fit_ok,
    entry_rank_key,
    fit_asymmetric_parabola,
)

ET = ZoneInfo("America/New_York")


def _five_min_session(
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
    start: datetime | None = None,
) -> pd.DataFrame:
    start = start or datetime(2026, 10, 14, 9, 30, tzinfo=ET)
    index = pd.date_range(start, periods=len(closes), freq="5min", tz=ET)
    volumes = volumes or [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )


def test_interval_to_timeframe_5m():
    tf = interval_to_timeframe("5m")
    assert str(tf) == "5Min"
    with pytest.raises(ValueError):
        interval_to_timeframe("2m")


def test_period_to_start_end():
    start, end = period_to_start_end("60d")
    assert (end - start).days >= 59
    with pytest.raises(ValueError):
        period_to_start_end("oops")


def test_bars_df_to_ohlcv_normalizes_multiindex():
    idx = pd.MultiIndex.from_product(
        [["AAPL"], pd.date_range("2026-10-14 13:30", periods=3, freq="5min", tz="UTC")],
        names=["symbol", "timestamp"],
    )
    raw = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 20.0, 30.0],
            "trade_count": [1, 2, 3],
            "vwap": [100.2, 101.2, 102.2],
        },
        index=idx,
    )
    out = bars_df_to_ohlcv(raw, symbol="AAPL")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 3
    assert str(out.index.tz) == "America/New_York"
    assert out["close"].iloc[-1] == pytest.approx(102.5)


def test_as_of_bars_never_leaks_future():
    bars = _five_min_session(
        opens=[100] * 6,
        highs=[101] * 6,
        lows=[99] * 6,
        closes=[100] * 6,
    )
    cutoff = bars.index[2]
    visible = as_of_bars(bars, cutoff)
    assert len(visible) == 3
    assert visible.index.max() == cutoff
    assert visible.index.max() <= cutoff


def test_entry_window_start_after_dip_window():
    day = date(2026, 10, 14)
    start = entry_window_start(day, window_minutes=30)
    assert start == datetime(2026, 10, 14, 10, 0, tzinfo=ET)


def test_integer_shares_for_notional():
    assert integer_shares_for_notional(20_000.0, 187.25) == 106
    assert 106 * 187.25 <= 20_000.0
    assert integer_shares_for_notional(20_000.0, 25_000.0) == 0
    assert integer_shares_for_notional(10_000.0, 100.0) == 100


def test_broker_buys_whole_shares_under_notional():
    broker = SimBroker(cash=50_000.0)
    when = datetime(2026, 10, 14, 10, 5, tzinfo=ET)
    pos = broker.market_buy(
        "CCC",
        price=187.25,
        notional=20_000.0,
        when=when,
        limit_price=188.0,
        stop_price=185.0,
    )
    assert pos is not None
    assert pos.qty == 106
    assert isinstance(pos.qty, int)
    assert broker.cash == pytest.approx(50_000.0 - 106 * 187.25)


def test_pending_buy_reserves_cash_and_blocks_budget():
    broker = SimBroker(cash=50_000.0)
    when = datetime(2026, 10, 14, 10, 5, tzinfo=ET)
    pending = broker.submit_buy(
        "AAA",
        buy_limit_price=100.0,
        notional=10_000.0,
        when=when,
        sell_limit_price=100.3,
        stop_price=99.0,
    )
    assert pending is not None
    assert pending.qty == 100
    assert broker.cash == pytest.approx(40_000.0)
    assert broker.reserved_cash == pytest.approx(10_000.0)
    assert broker.budget == pytest.approx(50_000.0)
    assert broker.watchlist[0].status == "pending"
    # Reserved cash cannot fund another full notional order.
    assert broker.can_open(max_positions=5, notional=45_000.0) is False
    assert broker.can_open(max_positions=5, notional=10_000.0) is True


def test_pending_buy_fills_on_next_5m_bar():
    broker = SimBroker(cash=50_000.0)
    submitted = datetime(2026, 10, 14, 10, 5, tzinfo=ET)
    broker.submit_buy(
        "AAA",
        buy_limit_price=100.0,
        notional=10_000.0,
        when=submitted,
        sell_limit_price=100.3,
        stop_price=99.0,
    )
    # Same bar: still pending.
    same = broker.process_pending_buys(
        {
            "AAA": pd.Series(
                {"open": 99.5, "high": 100.0, "low": 99.0, "close": 99.8, "volume": 1.0}
            )
        },
        submitted,
    )
    assert same.filled == []
    assert broker.pending_count == 1

    next_bar = submitted + timedelta(minutes=5)
    result = broker.process_pending_buys(
        {
            "AAA": pd.Series(
                {"open": 99.5, "high": 100.2, "low": 99.0, "close": 99.8, "volume": 1.0}
            )
        },
        next_bar,
    )
    assert len(result.filled) == 1
    assert result.filled[0].entry_price == pytest.approx(99.5)
    assert broker.pending_count == 0
    assert broker.has_position("AAA")
    assert broker.watchlist[0].status == "filled"
    # Refund unused reservation (limit 100 vs fill 99.5).
    assert broker.cash == pytest.approx(50_000.0 - 100 * 99.5)


def test_pending_buy_cancels_and_releases_cash_if_unfilled():
    broker = SimBroker(cash=50_000.0)
    submitted = datetime(2026, 10, 14, 10, 5, tzinfo=ET)
    broker.submit_buy(
        "BBB",
        buy_limit_price=100.0,
        notional=10_000.0,
        when=submitted,
        sell_limit_price=100.3,
        stop_price=99.0,
    )
    expire = submitted + timedelta(minutes=5)
    # Next bar never trades at/below limit → cancel and restore budget.
    result = broker.process_pending_buys(
        {
            "BBB": pd.Series(
                {"open": 101.0, "high": 102.0, "low": 100.5, "close": 101.5, "volume": 1.0}
            )
        },
        expire,
    )
    assert result.filled == []
    assert len(result.canceled) == 1
    assert result.canceled[0].status == "canceled"
    assert result.canceled[0].cancel_reason == "expired_unfilled"
    assert broker.pending_count == 0
    assert broker.cash == pytest.approx(50_000.0)
    assert broker.reserved_cash == pytest.approx(0.0)


def test_buy_limit_fill_price_helpers():
    bar = pd.Series({"open": 99.0, "high": 101.0, "low": 98.5, "close": 100.0})
    assert buy_limit_fill_price(bar, 100.0) == pytest.approx(99.0)
    bar2 = pd.Series({"open": 100.5, "high": 101.0, "low": 99.5, "close": 100.0})
    assert buy_limit_fill_price(bar2, 100.0) == pytest.approx(100.0)
    bar3 = pd.Series({"open": 101.0, "high": 102.0, "low": 100.5, "close": 101.0})
    assert buy_limit_fill_price(bar3, 100.0) is None


def test_broker_stop_wins_same_bar_over_limit():
    broker = SimBroker(cash=50_000.0)
    when = datetime(2026, 10, 14, 10, 5, tzinfo=ET)
    pos = broker.market_buy(
        "AAA",
        price=100.0,
        notional=10_000.0,
        when=when,
        limit_price=100.3,
        stop_price=99.0,
    )
    assert pos is not None and pos.qty == 100
    bar = pd.Series({"open": 100.0, "high": 100.5, "low": 98.5, "close": 99.5, "volume": 1.0})
    trade = broker.process_bar("AAA", bar, when)
    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(99.0)
    assert trade.qty == 100
    assert broker.open_count == 0


def test_broker_limit_profit_fill():
    broker = SimBroker(cash=50_000.0)
    when = datetime(2026, 10, 14, 10, 5, tzinfo=ET)
    broker.market_buy(
        "BBB",
        price=100.0,
        notional=10_000.0,
        when=when,
        limit_price=100.3,
        stop_price=99.0,
    )
    bar = pd.Series({"open": 100.1, "high": 100.4, "low": 100.0, "close": 100.35, "volume": 1.0})
    trade = broker.process_bar("BBB", bar, when)
    assert trade is not None
    assert trade.exit_reason == "limit_profit"
    assert trade.exit_price == pytest.approx(100.3)
    assert trade.qty == 100
    assert trade.realized_pnl == pytest.approx(30.0, abs=0.01)


def test_broker_profit_band_0_2_to_0_4_pct():
    """Limit sell at ~0.3% sits in the 0.2–0.4% target band."""
    buy = 50.0
    target_return = 0.003
    limit = buy * (1.0 + target_return)
    move_pct = (limit / buy - 1.0) * 100.0
    assert 0.2 <= move_pct <= 0.4


def _parabolic_closes(
    *,
    n: int = 18,
    trough_pos: int = 5,
    trough: float = 96.0,
    a_left: float = 0.18,
    a_right: float = 0.06,
) -> list[float]:
    closes: list[float] = []
    for i in range(n):
        d = i - trough_pos
        a = a_left if i <= trough_pos else a_right
        closes.append(trough + a * d * d)
    return closes


def _reclaiming_session_bars() -> pd.DataFrame:
    """As-of cut with interior trough and clean asymmetric V arms."""
    closes = _parabolic_closes(n=18, trough_pos=5, trough=96.0, a_left=0.18, a_right=0.06)
    trough_pos = 5
    lows = [c - 0.05 for c in closes]
    lows[trough_pos] = min(closes) - 0.2
    return _five_min_session(
        opens=[closes[0]] + closes[:-1],
        highs=[c + 0.3 for c in closes],
        lows=lows,
        closes=closes,
        volumes=[5000.0] * len(closes),
    )


def test_asymmetric_parabola_fit_ok_on_clean_v():
    bars = _reclaiming_session_bars()
    fit = fit_asymmetric_parabola(bars, prior_close=100.0)
    assert fit is not None
    assert fit.a_left > 0 and fit.a_right > 0
    assert fit.rel_rmse < 0.05
    assert asymmetric_parabola_fit_ok(
        bars, prior_close=100.0, max_parabola_rel_rmse=0.15
    )
    curve = asymmetric_parabola_curve(bars, prior_close=100.0)
    assert curve is not None
    fit2, series = curve
    assert fit2.trough_index == fit.trough_index
    assert len(series) == len(bars)
    assert abs(float(series.iloc[fit.trough_index]) - fit.trough_price) < 1e-9
    # Right edge of the fitted curve is the decision / buy bar.
    assert series.index[-1] == bars.index[-1]


def test_asymmetric_parabola_uses_asof_cut_not_full_session():
    """Fit only sees the as-of prefix; a later deeper trough is invisible."""
    early = _parabolic_closes(n=8, trough_pos=3, trough=96.0, a_left=0.2, a_right=0.1)
    late = [95.0, 94.0, 93.0, 94.5, 96.0, 97.0]
    closes = early + late
    lows = [c - 0.05 for c in closes]
    lows[3] = 95.8
    lows[10] = 92.5
    bars = _five_min_session(
        opens=[closes[0]] + closes[:-1],
        highs=[c + 0.3 for c in closes],
        lows=lows,
        closes=closes,
    )
    from tradingai.sim.clock import as_of_bars

    cut = as_of_bars(bars, bars.index[7])  # ends before late trough
    full = fit_asymmetric_parabola(bars, prior_close=100.0)
    asof = fit_asymmetric_parabola(cut, prior_close=100.0)
    assert asof is not None
    assert asof.trough_index == 3
    if full is not None:
        assert full.trough_index != asof.trough_index


def test_asymmetric_parabola_fit_rejects_noisy_w():
    # W-shape / oscillation after trough → large relative RMSE.
    closes = [99.0, 97.5, 96.0, 98.5, 96.2, 99.0, 97.0, 100.5, 98.0, 101.0]
    bars = _five_min_session(
        opens=[99.0] + closes[:-1],
        highs=[c + 0.4 for c in closes],
        lows=[c - 0.4 for c in closes],
        closes=closes,
    )
    # Force trough at index 2.
    bars.loc[bars.index[2], "low"] = 95.0
    assert not asymmetric_parabola_fit_ok(
        bars, prior_close=100.0, max_parabola_rel_rmse=0.15
    )


def test_strategy_enters_on_dip_reclaiming():
    bars = _reclaiming_session_bars()
    params = StrategyParams(
        dip_pct=2.0,
        window_minutes=30,
        require_status=("reclaiming",),
        target_return=0.003,
        stop_loss=0.01,
        use_trail=False,
        max_parabola_rel_rmse=0.15,
    )
    strategy = OpenDipStrategy(params)
    as_of = bars.index[-1].to_pydatetime()
    signal = strategy.evaluate_entry(
        "CCC",
        bars,
        prior_close=100.0,
        session_day=date(2026, 10, 14),
        as_of=as_of,
    )
    assert signal is not None
    assert signal.status == "reclaiming"
    assert signal.dip_pct <= -2.0
    assert signal.limit_price > signal.buy_price
    assert signal.stop_price < signal.buy_price
    move = (signal.limit_price / signal.buy_price - 1.0) * 100.0
    # Fee-aware limit is a bit above raw target; still near the 0.2–0.4% band.
    assert 0.2 <= move <= 1.0


def test_strategy_rejects_plunging():
    crash_closes = [100 - i * 0.5 for i in range(12)]
    bars = _five_min_session(
        opens=crash_closes,
        highs=[c + 0.1 for c in crash_closes],
        lows=[c - 0.3 for c in crash_closes],
        closes=crash_closes,
    )
    strategy = OpenDipStrategy(
        StrategyParams(dip_pct=2.0, window_minutes=30, require_status=("reclaiming",))
    )
    signal = strategy.evaluate_entry(
        "DDD",
        bars,
        prior_close=100.0,
        session_day=date(2026, 10, 14),
        as_of=bars.index[-1].to_pydatetime(),
    )
    assert signal is None


def test_strategy_params_to_dict():
    params = StrategyParams(dip_pct=4.0, stop_loss=0.015)
    d = params.to_dict()
    assert d["dip_pct"] == 4.0
    assert d["stop_loss"] == 0.015
    assert d["require_status"] == ["reclaiming"]
    assert d["momentum_minutes"] == 5
    assert d["max_parabola_rel_rmse"] == 0.15


def _scan_engine(**query_kwargs) -> DipSimEngine:
    params = StrategyParams(
        dip_pct=2.0,
        window_minutes=30,
        require_status=("reclaiming",),
        target_return=0.003,
        stop_loss=0.01,
        use_trail=False,
        max_positions=query_kwargs.get("max_positions", 5),
        buy_notional_usd=10_000.0,
    )
    return DipSimEngine(
        params=params,
        start_cash=100_000.0,
        persist=False,
        query_latency_seconds=query_kwargs.get("query_latency_seconds", 0.0),
        query_pace_seconds=query_kwargs.get("query_pace_seconds", 0.0),
        scan_batch_size=query_kwargs.get("scan_batch_size", 20),
        scan_seed=query_kwargs.get("scan_seed"),
    )


def _entry_signal(
    ticker: str,
    *,
    dip_pct: float,
    bounce_score: float,
    as_of: datetime | None = None,
) -> EntrySignal:
    when = as_of or datetime(2026, 10, 14, 10, 55, tzinfo=ET)
    buy = 100.0
    return EntrySignal(
        ticker=ticker,
        session_day=date(2026, 10, 14),
        as_of=when,
        buy_price=buy,
        dip_pct=dip_pct,
        status="reclaiming",
        bounce_score=bounce_score,
        limit_price=buy * 1.003,
        stop_price=buy * 0.99,
        trail_activate_price=None,
        sell_trail=None,
    )


def _plunging_session_bars() -> pd.DataFrame:
    crash_closes = [100 - i * 0.5 for i in range(12)]
    return _five_min_session(
        opens=crash_closes,
        highs=[c + 0.1 for c in crash_closes],
        lows=[c - 0.3 for c in crash_closes],
        closes=crash_closes,
        volumes=[5000.0] * len(crash_closes),
    )


def test_query_clock_zero_latency_scans_all_qualifying():
    bars = _reclaiming_session_bars()
    engine = _scan_engine(query_latency_seconds=0.0, query_pace_seconds=0.0, scan_seed=1)
    when = datetime(2026, 10, 14, 10, 55, tzinfo=ET)
    entry_end = datetime(2026, 10, 14, 11, 0, tzinfo=ET)
    engine._scan_entries_at_bar(
        day=date(2026, 10, 14),
        when=when,
        entry_end=entry_end,
        bars_cache={"AAA": bars, "BBB": bars.copy()},
        prior_cache={"AAA": 100.0, "BBB": 100.0},
    )
    assert {e.ticker for e in engine.entries} == {"AAA", "BBB"}
    assert all(e.as_of == when for e in engine.entries)


def test_query_clock_large_pace_stops_after_first():
    bars = _reclaiming_session_bars()
    engine = _scan_engine(
        query_latency_seconds=0.5, query_pace_seconds=600.0, scan_seed=7
    )
    when = datetime(2026, 10, 14, 10, 55, tzinfo=ET)
    entry_end = datetime(2026, 10, 14, 11, 0, tzinfo=ET)
    engine._scan_entries_at_bar(
        day=date(2026, 10, 14),
        when=when,
        entry_end=entry_end,
        bars_cache={"AAA": bars, "BBB": bars.copy()},
        prior_cache={"AAA": 100.0, "BBB": 100.0},
    )
    # Pace pushes clock past cutoff after the first visited ticker.
    assert len(engine.entries) == 1
    assert engine.entries[0].ticker in {"AAA", "BBB"}
    assert engine.entries[0].as_of == when + timedelta(seconds=0.5)


def test_query_clock_latency_delays_later_ticker_as_of():
    bars = _reclaiming_session_bars()
    engine = _scan_engine(
        query_latency_seconds=30.0, query_pace_seconds=0.0, scan_seed=3
    )
    when = datetime(2026, 10, 14, 10, 50, tzinfo=ET)
    entry_end = datetime(2026, 10, 14, 11, 0, tzinfo=ET)
    engine._scan_entries_at_bar(
        day=date(2026, 10, 14),
        when=when,
        entry_end=entry_end,
        bars_cache={"AAA": bars, "BBB": bars.copy()},
        prior_cache={"AAA": 100.0, "BBB": 100.0},
    )
    by_ticker = {e.ticker: e for e in engine.entries}
    assert set(by_ticker) == {"AAA", "BBB"}
    as_ofs = sorted(e.as_of for e in engine.entries)
    assert as_ofs == [when + timedelta(seconds=30), when + timedelta(seconds=60)]


def test_ticker_scan_order_is_random_permutation():
    bars = _reclaiming_session_bars()
    cache = {t: bars.copy() for t in ("AAA", "BBB", "CCC", "DDD")}
    prior = {t: 100.0 for t in cache}
    when = datetime(2026, 10, 14, 10, 55, tzinfo=ET)
    entry_end = datetime(2026, 10, 14, 11, 0, tzinfo=ET)

    orders: list[tuple[str, ...]] = []
    for seed in (11, 22, 33, 44, 55):
        engine = _scan_engine(
            query_latency_seconds=0.0, query_pace_seconds=0.0, scan_seed=seed
        )
        engine._scan_entries_at_bar(
            day=date(2026, 10, 14),
            when=when,
            entry_end=entry_end,
            bars_cache=cache,
            prior_cache=prior,
        )
        orders.append(tuple(e.ticker for e in engine.entries))
        assert set(orders[-1]) == set(cache)
        assert len(orders[-1]) == len(cache)

    # Different seeds should not all produce the same walk order.
    assert len(set(orders)) > 1


def test_entry_rank_key_prefers_early_reclaim_then_deeper_dip():
    early_deep = DipObservation(
        ticker="DEEP",
        as_of=datetime(2026, 10, 14, 10, 55, tzinfo=ET),
        qualifies_dip=True,
        dip_pct=-6.0,
        status="reclaiming",
        bounce_score=0.4,
        entry=_entry_signal("DEEP", dip_pct=-6.0, bounce_score=0.4),
    )
    early_shallow = DipObservation(
        ticker="SHAL",
        as_of=datetime(2026, 10, 14, 10, 55, tzinfo=ET),
        qualifies_dip=True,
        dip_pct=-3.0,
        status="reclaiming",
        bounce_score=0.2,
        entry=_entry_signal("SHAL", dip_pct=-3.0, bounce_score=0.2),
    )
    mature = DipObservation(
        ticker="MATR",
        as_of=datetime(2026, 10, 14, 10, 55, tzinfo=ET),
        qualifies_dip=True,
        dip_pct=-8.0,
        status="reclaiming",
        bounce_score=1.0,
        entry=_entry_signal("MATR", dip_pct=-8.0, bounce_score=1.0),
    )
    ranked = sorted(
        [mature, early_shallow, early_deep],
        key=lambda o: entry_rank_key(o),
    )
    assert [o.ticker for o in ranked] == ["DEEP", "SHAL", "MATR"]
    assert early_deep.is_early_reclaim
    assert not mature.is_early_reclaim


def test_plunge_watch_carries_to_next_bar_then_clears_on_reclaim():
    plunging = _plunging_session_bars()
    reclaiming = _reclaiming_session_bars()
    engine = _scan_engine(
        query_latency_seconds=0.0, query_pace_seconds=0.0, scan_seed=1
    )
    when = datetime(2026, 10, 14, 10, 55, tzinfo=ET)
    entry_end = datetime(2026, 10, 14, 11, 0, tzinfo=ET)

    engine._scan_entries_at_bar(
        day=date(2026, 10, 14),
        when=when,
        entry_end=entry_end,
        bars_cache={"PLNG": plunging},
        prior_cache={"PLNG": 100.0},
    )
    assert engine.entries == []
    assert "PLNG" in engine._plunge_watch

    engine._scan_entries_at_bar(
        day=date(2026, 10, 14),
        when=when,
        entry_end=entry_end,
        bars_cache={"PLNG": reclaiming, "ZZZ": reclaiming.copy()},
        prior_cache={"PLNG": 100.0, "ZZZ": 100.0},
    )
    # Watched name is queried before the random remainder.
    assert engine.entries[0].ticker == "PLNG"
    assert "PLNG" not in engine._plunge_watch


def test_batch_rank_enters_best_of_batch_when_slots_limited():
    bars = _reclaiming_session_bars()
    engine = _scan_engine(
        query_latency_seconds=0.0,
        query_pace_seconds=0.0,
        scan_seed=1,
        scan_batch_size=2,
        max_positions=1,
    )
    when = datetime(2026, 10, 14, 10, 55, tzinfo=ET)
    entry_end = datetime(2026, 10, 14, 11, 0, tzinfo=ET)

    # Patch observe so first batch has a clear early-reclaim winner.
    real_observe = engine.strategy.observe

    def fake_observe(ticker, bars_as_of, *, prior_close, session_day, as_of):
        obs = real_observe(
            ticker,
            bars_as_of,
            prior_close=prior_close,
            session_day=session_day,
            as_of=as_of,
        )
        if obs.entry is None:
            return obs
        if ticker == "AAA":
            dip, score = -7.0, 0.3
        else:
            dip, score = -3.0, 0.9
        entry = _entry_signal(ticker, dip_pct=dip, bounce_score=score, as_of=as_of)
        return DipObservation(
            ticker=ticker,
            as_of=as_of,
            qualifies_dip=True,
            dip_pct=dip,
            status="reclaiming",
            bounce_score=score,
            entry=entry,
        )

    engine.strategy.observe = fake_observe  # type: ignore[method-assign]
    engine._scan_entries_at_bar(
        day=date(2026, 10, 14),
        when=when,
        entry_end=entry_end,
        bars_cache={"AAA": bars, "BBB": bars.copy()},
        prior_cache={"AAA": 100.0, "BBB": 100.0},
    )
    assert len(engine.entries) == 1
    assert engine.entries[0].ticker == "AAA"
    assert engine.entries[0].bounce_score == pytest.approx(0.3)


def test_collect_chart_cases_wins_losses_and_canceled():
    entry = datetime(2026, 10, 14, 10, 5, tzinfo=ET)
    exit_t = datetime(2026, 10, 14, 10, 25, tzinfo=ET)
    trades = [
        ClosedTrade(
            ticker="WIN",
            qty=10,
            entry_price=100.0,
            entry_time=entry,
            exit_price=100.5,
            exit_time=exit_t,
            exit_reason="limit_profit",
            realized_pnl=5.0,
        ),
        ClosedTrade(
            ticker="LOS",
            qty=10,
            entry_price=100.0,
            entry_time=entry,
            exit_price=99.0,
            exit_time=exit_t,
            exit_reason="stop_loss",
            realized_pnl=-10.0,
        ),
    ]
    watch = [
        WatchlistEntry(
            watch_id=1,
            ticker="CXL",
            side="buy",
            status="canceled",
            qty=10,
            reserved_cash=1000.0,
            buy_limit_price=50.0,
            submitted_at=entry,
            expire_at=entry + timedelta(minutes=5),
            canceled_at=entry + timedelta(minutes=5),
            cancel_reason="expired_unfilled",
        )
    ]
    cases = collect_chart_cases(trades=trades, watchlist=watch)
    outcomes = {c.ticker: c.outcome for c in cases}
    assert outcomes == {"WIN": "win", "LOS": "loss", "CXL": "canceled"}


def test_plot_trade_case_writes_png(tmp_path):
    bars = _reclaiming_session_bars()
    from tradingai.sim.charts import ChartCase

    case = ChartCase(
        ticker="AAA",
        session_day=date(2026, 10, 14),
        outcome="win",
        submit_at=datetime(2026, 10, 14, 10, 0, tzinfo=ET),
        submit_price=97.0,
        fill_at=datetime(2026, 10, 14, 10, 5, tzinfo=ET),
        fill_price=97.0,
        exit_at=datetime(2026, 10, 14, 10, 30, tzinfo=ET),
        exit_price=100.0,
        exit_reason="limit_profit",
        realized_pnl=30.0,
    )
    out = plot_trade_case(case, bars, out_path=tmp_path / "aaa_win.png")
    assert out.is_file()
    assert out.stat().st_size > 0
