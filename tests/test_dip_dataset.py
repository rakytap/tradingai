from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tradingai.intraday.dip_charts import (
    pick_stratified_samples,
    plot_dip_sample,
    write_dip_dataset_viz,
)
from tradingai.intraday.dip_dataset import (
    assemble_cuts,
    build_cut,
    count_level_crossings,
    find_dip_anchor,
    label_cut,
    recent_session_window,
    sample_cut_ends,
    target_sell_price_for_entry,
    write_dip_dataset,
)
from tradingai.intraday.jobs.backfill_intraday_bars import (
    last_completed_session_day,
    weekday_sessions,
)
from tradingai.intraday.jobs.build_dip_dataset import (
    resolve_prior_close,
    resolve_sim_prior_close,
    take_recent_sessions,
)
from tradingai.providers.ibkr_tws import format_ib_end_datetime

ET = ZoneInfo("America/New_York")


def _bars(
    lows: list[float],
    closes: list[float],
    *,
    highs: list[float] | None = None,
    start: datetime | None = None,
) -> pd.DataFrame:
    start = start or datetime(2026, 10, 14, 9, 30, tzinfo=ET)
    index = pd.date_range(start, periods=len(closes), freq="1min", tz=ET)
    high = highs if highs is not None else [max(lo, cl) for lo, cl in zip(lows, closes)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": high,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=index,
    )


def _recovery_session() -> pd.DataFrame:
    """Decline through 95, bounce to 110, then stall just above 100."""
    closes: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for i in range(20):
        close = 100.0 - i * 0.1
        closes.append(close)
        lows.append(close)
        highs.append(close)
    closes.append(95.5)
    lows.append(95.0)
    highs.append(96.0)
    for _ in range(10):
        closes.append(95.5)
        lows.append(95.0)
        highs.append(95.6)
    closes.append(100.0)
    lows.append(95.5)
    highs.append(110.0)
    for _ in range(18):
        closes.append(100.0)
        lows.append(99.9)
        highs.append(100.05)
    return _bars(lows, closes, highs=highs)


def _valley_session(
    *,
    early_bottom: float = 97.0,
    mid_bottom: float = 95.0,
    late_bottom: float = 90.0,
    n_morning: int = 100,
    n_afternoon: int = 50,
) -> pd.DataFrame:
    """Two morning valleys then a deeper afternoon print outside the window."""
    closes: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    # Climb / shallow valley around bar 20.
    for i in range(40):
        if i < 20:
            close = 100.0 - (100.0 - early_bottom) * (i / 20.0)
        else:
            close = early_bottom + (100.0 - early_bottom) * ((i - 20) / 20.0)
        closes.append(close)
        lows.append(close - 0.05)
        highs.append(close + 0.05)
    lows[20] = early_bottom
    # Deeper, more pronounced valley around bar 60.
    for i in range(40, n_morning):
        if i < 60:
            close = 100.0 - (100.0 - mid_bottom) * ((i - 40) / 20.0)
        else:
            close = mid_bottom + (100.0 - mid_bottom) * ((i - 60) / 20.0)
        closes.append(close)
        lows.append(close - 0.05)
        highs.append(close + 0.05)
    lows[60] = mid_bottom
    # Afternoon dump deeper than either morning valley — must not become the dip.
    for i in range(n_afternoon):
        close = 100.0 - (100.0 - late_bottom) * min(1.0, i / 20.0)
        closes.append(close)
        lows.append(min(close, late_bottom) if i >= 20 else close - 0.05)
        highs.append(close + 0.05)
    lows[n_morning + 25] = late_bottom
    return _bars(lows, closes, highs=highs)


def test_find_dip_anchor_picks_most_pronounced_morning_valley():
    bars = _valley_session()
    anchor = find_dip_anchor(
        bars,
        prior_close=100.0,
        dip_pct=4.0,
        valley_window_minutes=90,
        valley_order_minutes=5,
    )
    assert anchor is not None
    assert anchor.anchor_time == bars.index[60]
    assert anchor.anchor_low == 95.0
    assert anchor.dip_pct == -5.0
    # Afternoon low is deeper but outside the valley window.
    assert float(bars["low"].min()) == 90.0


def test_find_dip_anchor_ignores_session_wide_low_after_window():
    bars = _valley_session(mid_bottom=95.5, late_bottom=80.0)
    anchor = find_dip_anchor(bars, prior_close=100.0, dip_pct=4.0)
    assert anchor is not None
    assert anchor.anchor_time < bars.index[90]
    assert anchor.anchor_low > 80.0


def test_find_dip_anchor_rejects_shallow_morning():
    bars = _bars(
        lows=[99.0] * 40 + [98.5] + [99.0] * 40,
        closes=[99.0] * 40 + [98.5] + [99.0] * 40,
    )
    assert find_dip_anchor(bars, prior_close=100.0, dip_pct=4.0) is None


def test_find_dip_anchor_prefers_open_washout_by_right_bounce():
    """CHWY-like: sharp open dip with tiny left side beats a later shallower valley."""
    closes: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    # Bar 0 open, bar 2 washout to 91, quick bounce to 98.
    path = [100.0, 96.0, 91.0, 94.0, 97.0, 98.0]
    for close in path:
        closes.append(close)
        lows.append(close - 0.05)
        highs.append(close + 0.05)
    lows[2] = 91.0
    # Later mid-morning valley to 94 with a smaller right bounce.
    for i in range(6, 90):
        if i < 40:
            close = 98.0
        elif i < 50:
            close = 98.0 - (98.0 - 94.0) * ((i - 40) / 10.0)
        else:
            close = 94.0 + (97.0 - 94.0) * min(1.0, (i - 50) / 15.0)
        closes.append(close)
        lows.append(close - 0.05)
        highs.append(close + 0.05)
    lows[45] = 94.0
    bars = _bars(lows, closes, highs=highs)
    anchor = find_dip_anchor(
        bars,
        prior_close=100.0,
        dip_pct=4.0,
        valley_window_minutes=90,
        valley_order_minutes=5,
    )
    assert anchor is not None
    assert anchor.anchor_time == bars.index[2]
    assert anchor.anchor_low == 91.0


def test_sample_cut_ends_in_session_window_and_reproducible():
    # 180 minutes from 09:30 → window [10:30, 12:00] = indices 60..150
    bars = _bars(lows=[100.0] * 180, closes=[100.0] * 180)
    anchor = bars.index[90]  # 11:00
    horizon = 15
    first = sample_cut_ends(
        bars,
        anchor,
        8,
        horizon,
        np.random.default_rng(0),
        cut_min_minutes_from_open=60,
        cut_max_minutes_from_open=150,
    )
    second = sample_cut_ends(
        bars,
        anchor,
        8,
        horizon,
        np.random.default_rng(0),
        cut_min_minutes_from_open=60,
        cut_max_minutes_from_open=150,
    )
    assert first == second
    assert len(first) == 8
    open_ts = bars.index[0]
    earliest = open_ts + pd.Timedelta(minutes=60)
    latest = open_ts + pd.Timedelta(minutes=150)
    last_eligible = bars.index[len(bars) - horizon - 1]
    for stamp, side in first:
        assert stamp != anchor
        assert earliest <= stamp <= latest
        assert stamp <= last_eligible
        assert side == ("before" if stamp < anchor else "after")


def test_sample_cut_ends_takes_entire_pool_when_short():
    bars = _bars(lows=[100.0] * 180, closes=[100.0] * 180)
    anchor = bars.index[90]
    # Narrow window with only a few eligible bars.
    chosen = sample_cut_ends(
        bars,
        anchor,
        20,
        5,
        np.random.default_rng(1),
        cut_min_minutes_from_open=60,
        cut_max_minutes_from_open=65,
    )
    open_ts = bars.index[0]
    pool = [
        ts
        for ts in bars.index[: 180 - 5]
        if open_ts + pd.Timedelta(minutes=60)
        <= ts
        <= open_ts + pd.Timedelta(minutes=65)
        and ts != anchor
    ]
    assert {ts for ts, _ in chosen} == set(pool)


def test_sample_cut_ends_excludes_early_and_late_session_bars():
    bars = _bars(lows=[100.0] * 200, closes=[100.0] * 200)
    anchor = bars.index[100]
    chosen = sample_cut_ends(
        bars,
        anchor,
        40,
        5,
        np.random.default_rng(2),
        cut_min_minutes_from_open=60,
        cut_max_minutes_from_open=150,
    )
    assert chosen
    open_ts = bars.index[0]
    earliest = open_ts + pd.Timedelta(minutes=60)
    latest = open_ts + pd.Timedelta(minutes=150)
    for stamp, side in chosen:
        assert stamp != anchor
        assert earliest <= stamp <= latest
        assert side == ("before" if stamp < anchor else "after")
    assert all(stamp >= earliest for stamp, _ in chosen)
    assert all(stamp <= latest for stamp, _ in chosen)


def test_build_cut_starts_with_prior_close_and_stops_at_cut_end():
    bars = _recovery_session()
    cut_end = bars.index[3]
    cut = build_cut(bars, prior_close=100.0, cut_end=cut_end)
    assert bool(cut.iloc[0]["is_prior_close"]) is True
    assert float(cut.iloc[0]["close"]) == 100.0
    assert float(cut.iloc[0]["volume"]) == 0.0
    assert cut.index[-1] == cut_end
    assert bool(cut.iloc[-1]["is_prior_close"]) is False
    assert len(cut) == 5
    assert cut.index[-1] <= cut_end


def test_label_cut_before_dip_is_always_negative():
    bars = _recovery_session()
    dip = bars.index[20]
    pre_end = bars.index[10]
    # Even if a later high would clear a sell target, plunging-side cuts are -1.
    label, _high = label_cut(
        bars,
        pre_end,
        entry_close=float(bars.loc[pre_end, "close"]),
        side="before",
        anchor_time=dip,
    )
    assert label == -1
    label_by_anchor, _ = label_cut(
        bars,
        pre_end,
        entry_close=float(bars.loc[pre_end, "close"]),
        anchor_time=dip,
    )
    assert label_by_anchor == -1


def test_label_cut_after_dip_sell_target_within_horizon_is_positive():
    bars = _recovery_session()
    # First bar at the trough (low 95); within 30m a high of 110 clears the sell tgt.
    trough_end = bars.index[20]
    entry = float(bars.loc[trough_end, "close"])
    sell = target_sell_price_for_entry(entry)
    label, max_high = label_cut(
        bars,
        trough_end,
        entry_close=entry,
        side="after",
        anchor_time=trough_end,
        sell_target=sell,
        label_horizon_minutes=30,
    )
    assert max_high == 110.0
    assert max_high >= sell
    assert label == 1


def test_label_cut_after_dip_missed_sell_target_is_zero():
    bars = _recovery_session()
    late_end = bars.index[40]
    late_close = float(bars.loc[late_end, "close"])
    sell = target_sell_price_for_entry(late_close)
    label, max_high = label_cut(
        bars,
        late_end,
        entry_close=late_close,
        side="after",
        anchor_time=bars.index[20],
        sell_target=sell,
        label_horizon_minutes=30,
    )
    assert late_close == 100.0
    assert max_high == 100.05
    assert max_high < sell
    assert label == 0


def test_label_cut_sell_target_after_horizon_is_zero():
    bars = _recovery_session()
    trough_end = bars.index[20]
    entry = float(bars.loc[trough_end, "close"])
    sell = target_sell_price_for_entry(entry)
    # Spike to 110 is ~11 minutes after the trough; a 5-minute horizon misses it.
    label, max_high = label_cut(
        bars,
        trough_end,
        entry_close=entry,
        side="after",
        sell_target=sell,
        label_horizon_minutes=5,
    )
    assert max_high is not None and max_high < sell
    assert label == 0


def test_label_cut_stop_first_then_target_is_zero():
    """Stop hit before target within the horizon → label 0."""
    closes = [100.0, 99.2, 101.0]
    lows = [100.0, 99.2, 100.5]
    highs = [100.0, 99.5, 101.0]
    bars = _bars(lows, closes, highs=highs)
    cut_end = bars.index[0]
    entry = 100.0
    sell = target_sell_price_for_entry(entry)
    assert highs[1] < sell
    assert highs[2] >= sell
    assert lows[1] <= entry * (1.0 - 0.006)
    label, max_high = label_cut(
        bars,
        cut_end,
        entry_close=entry,
        side="after",
        sell_target=sell,
        label_horizon_minutes=15,
        stop_loss=0.006,
    )
    assert max_high == 101.0
    assert label == 0


def test_label_cut_target_first_then_stop_is_one():
    """Target prints before any stop touch → label 1 (later stop ignored)."""
    closes = [100.0, 101.0, 99.0]
    lows = [100.0, 100.5, 99.0]
    highs = [100.0, 101.0, 99.5]
    bars = _bars(lows, closes, highs=highs)
    cut_end = bars.index[0]
    entry = 100.0
    sell = target_sell_price_for_entry(entry)
    assert highs[1] >= sell
    assert lows[1] > entry * (1.0 - 0.006)
    assert lows[2] <= entry * (1.0 - 0.006)
    label, _ = label_cut(
        bars,
        cut_end,
        entry_close=entry,
        side="after",
        sell_target=sell,
        label_horizon_minutes=15,
        stop_loss=0.006,
    )
    assert label == 1


def test_label_cut_without_stop_hit_can_still_be_positive():
    closes = [100.0, 99.5, 101.0]
    lows = [100.0, 99.5, 100.5]
    highs = [100.0, 99.7, 101.0]
    bars = _bars(lows, closes, highs=highs)
    cut_end = bars.index[0]
    entry = 100.0
    sell = target_sell_price_for_entry(entry)
    assert lows[1] > entry * (1.0 - 0.006)
    assert highs[2] >= sell
    label, _ = label_cut(
        bars,
        cut_end,
        entry_close=entry,
        side="after",
        sell_target=sell,
        label_horizon_minutes=15,
        stop_loss=0.006,
    )
    assert label == 1


def test_label_defaults_read_from_env(monkeypatch):
    from tradingai.intraday.config import clear_intraday_config_cache

    monkeypatch.setenv("TRADINGAI_DIP_LABEL_STOP_LOSS", "0.01")
    monkeypatch.setenv("TRADINGAI_DIP_LABEL_HORIZON_MINUTES", "12")
    monkeypatch.setenv("TRADINGAI_OPEN_DIP_TARGET_RETURN", "0.01")
    monkeypatch.setenv("TRADINGAI_OPEN_DIP_BUY_NOTIONAL", "10000")
    clear_intraday_config_cache()
    from tradingai.intraday.config import (
        get_buy_notional_usd,
        get_label_horizon_minutes,
        get_label_stop_loss,
        get_target_return,
    )

    assert get_label_stop_loss() == 0.01
    assert get_label_horizon_minutes() == 12
    assert get_target_return() == 0.01
    assert get_buy_notional_usd() == 10000.0
    sell = target_sell_price_for_entry(100.0)
    assert sell > 100.0
    clear_intraday_config_cache()


def test_count_level_crossings_close_to_close():
    bars = _bars(
        lows=[100.0, 101.0, 99.0, 101.5, 98.0],
        closes=[100.0, 101.0, 99.0, 101.5, 98.0],
    )
    # Upward only: 100→101 and 99→101.5 (downward 101→99 / 101.5→98 ignored)
    assert count_level_crossings(bars, 100.5) == 2
    assert count_level_crossings(bars, 200.0) == 0


def test_target_sell_price_above_entry():
    entry = 100.0
    sell_at = target_sell_price_for_entry(entry)
    assert sell_at > entry


def test_take_recent_sessions_keeps_last_n():
    frames = [
        (date(2026, 10, d), pd.DataFrame({"close": [float(d)]}))
        for d in range(10, 16)
    ]
    assert take_recent_sessions(frames, None) == frames
    assert take_recent_sessions(frames, 0) == frames
    kept = take_recent_sessions(frames, 2)
    assert [day for day, _ in kept] == [date(2026, 10, 14), date(2026, 10, 15)]
    assert take_recent_sessions(frames, 100) == frames


def test_assemble_cuts_records_sell_crossings():
    day0 = date(2026, 10, 13)
    day1 = date(2026, 10, 14)
    # Prior session oscillates across ~100.4 so crossings accumulate.
    prior = _bars(
        lows=[100.0, 101.0, 99.5, 101.2, 99.0] + [100.0] * 35,
        closes=[100.0, 101.0, 99.5, 101.2, 99.0] + [100.0] * 35,
        start=datetime(2026, 10, 13, 9, 30, tzinfo=ET),
    )
    bars = _recovery_session()
    samples, _cut_bars, qualifying = assemble_cuts(
        [
            ("XYZ", day0, prior, 100.0),
            ("XYZ", day1, bars, 100.0),
        ],
        dip_pct=4.0,
        rise_pct=0.4,
        n_cuts=2,
        cut_min_minutes_from_open=0,
        cut_max_minutes_from_open=0,
        min_horizon_minutes=5,
        label_horizon_minutes=5,
        crossing_sessions=5,
        seed=0,
    )
    assert qualifying == 1  # only day1 hits the dip
    assert "sell_crossings_5d" in samples.columns
    assert "target_sell_price" in samples.columns
    assert "stop_loss_price" in samples.columns
    assert (samples["target_sell_price"] > samples["entry_close"]).all()
    assert (samples["stop_loss_price"] < samples["entry_close"]).all()
    assert (samples["sell_crossings_5d"] >= 0).all()
    # Window helper truncates the current session at cut_end.
    cut_end = samples.iloc[0]["cut_end"]
    window = recent_session_window([prior, bars], cut_end, max_sessions=5)
    assert window.index.max() <= cut_end


def test_assemble_cuts_labels_match_future_high_and_keep_the_prefix():
    bars = _recovery_session()
    anchor = find_dip_anchor(bars, prior_close=100.0, dip_pct=4.0)
    assert anchor is not None
    assert anchor.anchor_time == bars.index[20]
    samples, cut_bars, qualifying = assemble_cuts(
        [("abc", date(2026, 10, 14), bars, 100.0)],
        dip_pct=4.0,
        rise_pct=0.4,
        n_cuts=6,
        cut_min_minutes_from_open=0,
        cut_max_minutes_from_open=0,
        min_horizon_minutes=5,
        label_horizon_minutes=5,
        seed=2,
    )
    assert qualifying == 1
    assert len(samples) == 6
    assert set(samples["side"]) == {"before", "after"}
    assert samples["ticker"].tolist() == ["ABC"] * 6
    for row in samples.itertuples(index=False):
        expected, _high = label_cut(
            bars,
            row.cut_end,
            row.entry_close,
            side=row.side,
            anchor_time=anchor.anchor_time,
            sell_target=float(row.target_sell_price),
            label_horizon_minutes=5,
        )
        assert row.label == expected
        if row.side == "before":
            assert row.label == -1
        else:
            assert row.label in (0, 1)
        assert row.cut_end != anchor.anchor_time
        window = cut_bars[cut_bars["sample_id"] == row.sample_id].sort_values("seq")
        assert bool(window.iloc[0]["is_prior_close"]) is True
        assert float(window.iloc[0]["close"]) == 100.0
        assert window.iloc[-1]["bar_time"] == row.cut_end
        assert bool(window.iloc[-1]["is_prior_close"]) is False
        assert (window.iloc[1:]["bar_time"] <= row.cut_end).all()


def test_write_dip_dataset_roundtrip(tmp_path):
    bars = _recovery_session()
    samples, cut_bars, _qualifying = assemble_cuts(
        [("AAA", date(2026, 10, 14), bars, 100.0)],
        n_cuts=2,
        cut_min_minutes_from_open=0,
        cut_max_minutes_from_open=0,
        min_horizon_minutes=5,
        label_horizon_minutes=5,
        seed=0,
    )
    out = write_dip_dataset(samples, cut_bars, tmp_path / "dip_dataset")
    loaded_samples = pd.read_parquet(out / "samples.parquet")
    loaded_bars = pd.read_parquet(out / "bars.parquet")
    assert list(loaded_samples["label"]) == list(samples["label"])
    assert loaded_bars["is_prior_close"].iloc[0]
    assert len(loaded_bars) == len(cut_bars)


def test_resolve_prior_close_uses_daily_then_previous_session():
    daily = [(date(2026, 10, 13), 50.0), (date(2026, 10, 14), 99.0)]
    history = _bars(
        lows=[77.0],
        closes=[77.0],
        start=datetime(2026, 10, 13, 15, 59, tzinfo=ET),
    )
    assert resolve_prior_close("AAA", date(2026, 10, 15), daily, history) == 99.0
    assert resolve_prior_close("AAA", date(2026, 10, 15), [], history) == 77.0


def test_resolve_sim_prior_close_falls_back_to_previous_session(monkeypatch):
    history = _bars(
        lows=[77.0],
        closes=[77.0],
        start=datetime(2026, 10, 13, 15, 59, tzinfo=ET),
    )

    def _no_sim_prior(ticker, session_day, *, database_url=None):
        return None

    monkeypatch.setattr(
        "tradingai.intraday.jobs.build_dip_dataset.load_sim_prior_close",
        _no_sim_prior,
    )
    assert resolve_sim_prior_close("AAA", date(2026, 10, 15), history) == 77.0


def test_weekday_sessions_and_last_completed_day():
    assert weekday_sessions(3, end=date(2026, 10, 16)) == [
        date(2026, 10, 14),
        date(2026, 10, 15),
        date(2026, 10, 16),
    ]
    assert weekday_sessions(1, end=date(2026, 10, 18)) == [date(2026, 10, 16)]
    monday_morning = datetime(2026, 10, 19, 10, 0, tzinfo=ET)
    monday_close = datetime(2026, 10, 19, 16, 0, tzinfo=ET)
    saturday = datetime(2026, 10, 17, 12, 0, tzinfo=ET)
    assert last_completed_session_day(monday_morning) == date(2026, 10, 16)
    assert last_completed_session_day(monday_close) == date(2026, 10, 19)
    assert last_completed_session_day(saturday) == date(2026, 10, 16)


def test_backfill_skips_stored_days_and_retries_pacing(monkeypatch):
    from tradingai.data.repositories.intraday_bars import INTRADAY_HISTORY_TABLE
    from tradingai.intraday.jobs import backfill_intraday_bars as job

    monkeypatch.setattr(job, "initialize_schema", lambda: None)
    monkeypatch.setattr(job, "last_completed_session_day", lambda now=None: date(2026, 10, 14))
    monkeypatch.setattr(job, "log_start", lambda *a, **k: None)
    monkeypatch.setattr(job, "log_progress", lambda *a, **k: None)
    monkeypatch.setattr(job, "log_error", lambda *a, **k: None)
    monkeypatch.setattr(job, "log_done", lambda *a, **k: None)

    class FakeIB:
        RaiseRequestErrors = False

    class FakeConnection:
        def __enter__(self):
            return FakeIB()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(job, "connect_ib", lambda: FakeConnection())
    monkeypatch.setattr(job, "qualify_equity", lambda ib, ticker: "contract")
    monkeypatch.setattr(
        job,
        "session_has_bars",
        lambda ticker, day, *, table: day == date(2026, 10, 13),
    )

    calls: list[datetime] = []
    attempts = {"n": 0}

    def fetch(ib, ticker, *, period=None, interval=None, end_datetime=None, contract=None):
        calls.append(end_datetime)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("pacing violation")
        day = end_datetime.date()
        index = pd.date_range(
            datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET),
            periods=2,
            freq="1min",
        )
        return pd.DataFrame(
            {
                "open": [1.0, 1.1],
                "high": [1.2, 1.2],
                "low": [0.9, 1.0],
                "close": [1.1, 1.15],
                "volume": [10.0, 11.0],
            },
            index=index,
        )

    upserts: list[tuple] = []
    monkeypatch.setattr(job, "fetch_intraday_ohlcv_ibkr", fetch)
    monkeypatch.setattr(
        job,
        "upsert_intraday_bars",
        lambda ticker, rows, *, table: upserts.append((ticker, table, len(rows))) or len(rows),
    )

    result = job.backfill_intraday_bars(["aaa"], days=2, pace_seconds=0, max_retries=2)
    assert result["skipped"] == 1
    assert result["stored_days"] == 1
    assert result["bars_written"] == 2
    assert result["errors"] == []
    assert attempts["n"] == 2
    assert calls[0] == datetime(2026, 10, 14, 16, 0, tzinfo=ET)
    assert upserts == [("AAA", INTRADAY_HISTORY_TABLE, 2)]


def test_format_ib_end_datetime_converts_eastern_close_to_utc():
    close = datetime(2026, 10, 14, 16, 0, tzinfo=ET)
    assert format_ib_end_datetime(close) == "20261014 20:00:00 UTC"
    assert format_ib_end_datetime(datetime(2026, 10, 14, 16, 0)) == "20261014 20:00:00 UTC"
    assert format_ib_end_datetime(None) == ""
    assert format_ib_end_datetime("20261014 20:00:00 UTC") == "20261014 20:00:00 UTC"


def test_write_dip_dataset_viz_plots_full_session(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session = _recovery_session()
    samples, cut_bars, _ = assemble_cuts(
        [("AAA", date(2026, 10, 14), session, 100.0)],
        n_cuts=2,
        cut_min_minutes_from_open=0,
        cut_max_minutes_from_open=0,
        min_horizon_minutes=5,
        label_horizon_minutes=5,
        seed=0,
    )
    assert not samples.empty
    out = write_dip_dataset_viz(
        samples,
        cut_bars,
        tmp_path / "viz",
        sessions={("AAA", date(2026, 10, 14)): session},
        n=min(2, len(samples)),
        seed=0,
    )
    pngs = sorted(out.glob("sample_*.png"))
    assert len(pngs) >= 1
    assert (out / f"overview_{len(pngs)}.png").exists()

    # Full session is longer than the cut window through cut_end.
    row = samples.iloc[0]
    cut = cut_bars[cut_bars["sample_id"] == row.sample_id]
    cut_real = cut[~cut["is_prior_close"].astype(bool)]
    assert len(session) > len(cut_real)

    fig, ax = plt.subplots()
    plot_dip_sample(ax, row, session_bars=session, cut_bars=cut)
    # Pre-cut (dark) and post-cut (grey) close lines both present.
    assert len(ax.get_lines()) >= 2
    legend_text = " ".join(
        str(t.get_text()) for t in (ax.get_legend().get_texts() if ax.get_legend() else [])
    )
    assert "stop" in legend_text.lower()
    assert "cut px" in legend_text.lower()
    # Cut time is a green point; no vertical divider labeled cut end.
    assert not any(
        (line.get_label() or "").lower() == "cut end" for line in ax.get_lines()
    )
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    session_span = (
        session.index[-1].to_pydatetime().timestamp()
        - session.index[0].to_pydatetime().timestamp()
    ) / 86400.0
    assert span >= session_span * 0.9
    plt.close(fig)

    picked = pick_stratified_samples(samples, n=1, seed=0)
    assert len(picked) == 1
