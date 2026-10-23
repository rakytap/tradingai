from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tradingai.intraday.dip_dataset import (
    assemble_cuts,
    build_cut,
    find_dip_anchor,
    label_cut,
    sample_cut_ends,
    write_dip_dataset,
)
from tradingai.intraday.jobs.backfill_intraday_bars import (
    last_completed_session_day,
    weekday_sessions,
)
from tradingai.intraday.jobs.build_dip_dataset import resolve_prior_close
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


def test_find_dip_anchor_is_first_bar_at_or_below_96():
    bars = _bars(
        lows=[99.0, 97.0, 96.0, 95.0],
        closes=[99.0, 97.0, 96.0, 95.0],
    )
    anchor = find_dip_anchor(bars, prior_close=100.0, dip_pct=4.0)
    assert anchor is not None
    assert anchor.anchor_time == bars.index[2]
    assert anchor.anchor_low == 96.0
    assert anchor.dip_pct == -4.0

    shallow = _bars(lows=[99.0, 97.0, 96.01], closes=[99.0, 97.0, 96.01])
    assert find_dip_anchor(shallow, prior_close=100.0, dip_pct=4.0) is None


def test_sample_cut_ends_avoids_anchor_and_horizon():
    bars = _bars(lows=[100.0] * 40, closes=[100.0] * 40)
    anchor = bars.index[20]
    horizon = 15
    first = sample_cut_ends(
        bars,
        anchor,
        n_before=4,
        n_after=10,
        min_horizon_minutes=horizon,
        rng=np.random.default_rng(0),
    )
    second = sample_cut_ends(
        bars,
        anchor,
        n_before=4,
        n_after=10,
        min_horizon_minutes=horizon,
        rng=np.random.default_rng(0),
    )
    assert first == second
    before = [stamp for stamp, side in first if side == "before"]
    after = [stamp for stamp, side in first if side == "after"]
    assert len(before) == 4
    assert len(after) == 4
    last_eligible = bars.index[len(bars) - horizon - 1]
    for stamp, _side in first:
        assert stamp != anchor
        assert stamp <= last_eligible
    assert all(stamp < anchor for stamp in before)
    assert all(stamp > anchor for stamp in after)


def test_sample_cut_ends_takes_every_bar_when_the_side_is_short():
    bars = _bars(lows=[100.0] * 30, closes=[100.0] * 30)
    anchor = bars.index[5]
    chosen = sample_cut_ends(
        bars,
        anchor,
        n_before=10,
        n_after=10,
        min_horizon_minutes=20,
        rng=np.random.default_rng(1),
    )
    before = [stamp for stamp, side in chosen if side == "before"]
    after = [stamp for stamp, side in chosen if side == "after"]
    assert before == list(bars.index[:5])
    assert after == list(bars.index[6:10])
    assert anchor not in before and anchor not in after


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


def test_label_cut_uses_later_high_versus_cut_end_close():
    bars = _recovery_session()
    # Still falling: later bars print a new low (95), even though a rise follows.
    pre_end = bars.index[0]
    pre_label, pre_high = label_cut(bars, pre_end, entry_close=100.0, rise_pct=0.4)
    assert pre_high == 110.0
    assert pre_label == -1

    # After the trough, no new low and only a tiny high → 0.
    late_end = bars.index[40]
    late_close = float(bars.loc[late_end, "close"])
    late_label, late_high = label_cut(bars, late_end, entry_close=late_close, rise_pct=0.4)
    assert late_close == 100.0
    assert late_high == 100.05
    assert late_label == 0


def test_label_cut_rise_after_trough_is_positive():
    bars = _recovery_session()
    # First bar at the trough (low 95); later highs reach 110 without a deeper low.
    trough_end = bars.index[20]
    label, max_high = label_cut(
        bars,
        trough_end,
        entry_close=float(bars.loc[trough_end, "close"]),
        rise_pct=0.4,
    )
    assert max_high == 110.0
    assert label == 1


def test_assemble_cuts_labels_match_future_high_and_keep_the_prefix():
    bars = _recovery_session()
    anchor = find_dip_anchor(bars, prior_close=100.0, dip_pct=4.0)
    assert anchor is not None
    assert anchor.anchor_time == bars.index[20]
    samples, cut_bars, qualifying = assemble_cuts(
        [("abc", date(2026, 10, 14), bars, 100.0)],
        dip_pct=4.0,
        rise_pct=0.4,
        cuts_before=3,
        cuts_after=3,
        min_horizon_minutes=5,
        seed=2,
    )
    assert qualifying == 1
    assert set(samples["side"]) == {"before", "after"}
    assert (samples["side"] == "before").sum() == 3
    assert (samples["side"] == "after").sum() == 3
    assert samples["ticker"].tolist() == ["ABC"] * 6
    for row in samples.itertuples(index=False):
        expected, _high = label_cut(bars, row.cut_end, row.entry_close, 0.4)
        assert row.label == expected
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
        cuts_before=1,
        cuts_after=1,
        min_horizon_minutes=5,
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
