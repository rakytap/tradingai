from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingai.intraday.bounce import assess_bounce
from tradingai.intraday.open_dip import detect_open_dip, is_making_lower_lows
from tradingai.intraday.scan import scan_open_dip_bounces

ET = ZoneInfo("America/New_York")


def _session_bars(
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
    start: datetime | None = None,
) -> pd.DataFrame:
    start = start or datetime(2026, 10, 14, 9, 30, tzinfo=ET)
    index = pd.date_range(start, periods=len(closes), freq="1min", tz=ET)
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


def test_detect_open_dip_qualifies_on_deep_trough():
    # Prior close 100; trough 97 → -3%
    bars = _session_bars(
        opens=[99.5] + [98.0] * 59,
        highs=[99.5] + [98.5] * 59,
        lows=[99.0] + [97.0] + [97.5] * 58,
        closes=[99.2] + [97.8] * 59,
    )
    signal = detect_open_dip(bars, ticker="AAA", prior_close=100.0, dip_pct=2.0, window_minutes=60)
    assert signal.qualifies is True
    assert signal.dip_pct == pytest.approx(-3.0, abs=1e-3)


def test_detect_open_dip_rejects_shallow_move():
    bars = _session_bars(
        opens=[100.0] * 30,
        highs=[100.5] * 30,
        lows=[99.5] * 30,
        closes=[100.0] * 30,
    )
    signal = detect_open_dip(bars, ticker="AAA", prior_close=100.0, dip_pct=2.0)
    assert signal.qualifies is False


def test_is_making_lower_lows():
    bars = _session_bars(
        opens=[100, 99, 98, 97, 96],
        highs=[100, 99, 98, 97, 96],
        lows=[99.5, 98.5, 97.5, 96.5, 95.5],
        closes=[99, 98, 97, 96, 95],
    )
    assert is_making_lower_lows(bars, lookback=5) is True


def test_assess_bounce_reclaiming_vs_plunging():
    prior = 100.0
    # Plunge then reclaim open + VWAP with positive momentum (red first bar)
    n = 40
    closes = [97.0] * 20 + [98.5] * 10 + [100.5] * 10
    bars = _session_bars(
        opens=[99.0] + closes[:-1],
        highs=[c + 0.3 for c in closes],
        lows=[c - 0.3 for c in closes],
        closes=closes,
        volumes=[2000.0] * n,
    )
    dip = detect_open_dip(bars, ticker="BBB", prior_close=prior, dip_pct=2.0)
    assert dip.qualifies
    bounce = assess_bounce(bars, dip)
    assert bounce.status in {"reclaiming", "stabilizing"}
    assert bounce.first_bar_green is False
    assert bounce.bounce_score > -0.2

    # Free-fall: continuing lower lows
    crash_closes = [100 - i * 0.4 for i in range(20)]
    crash = _session_bars(
        opens=crash_closes,
        highs=[c + 0.1 for c in crash_closes],
        lows=[c - 0.2 for c in crash_closes],
        closes=crash_closes,
    )
    dip2 = detect_open_dip(crash, ticker="CCC", prior_close=100.0, dip_pct=2.0)
    assert dip2.qualifies
    plunging = assess_bounce(crash, dip2)
    assert plunging.status == "plunging"
    assert plunging.bounce_score < 0


def test_assess_bounce_penalizes_green_first_bar():
    prior = 100.0
    n = 40
    # After minute 15 both paths match; only the opening 15m composite differs.
    later = [96.5] * 5 + [98.5] * 10 + [100.5] * 10
    # Red 15m: session open 100 → close at 15m = 97
    red_closes = [100.0 - (100.0 - 97.0) * (i + 1) / 15 for i in range(15)] + later
    red = _session_bars(
        opens=[100.0] + red_closes[:-1],
        highs=[c + 0.3 for c in red_closes],
        lows=[c - 0.3 for c in red_closes],
        closes=red_closes,
        volumes=[2000.0] * n,
    )
    # Green 15m: session open 96 → close at 15m = 97.5, then same later path
    green_closes = [96.0 + (97.5 - 96.0) * (i + 1) / 15 for i in range(15)] + later
    green = _session_bars(
        opens=[96.0] + green_closes[:-1],
        highs=[c + 0.3 for c in green_closes],
        lows=[c - 0.3 for c in green_closes],
        closes=green_closes,
        volumes=[2000.0] * n,
    )
    # Green first 1m alone must not trigger: minute-0 green but 15m composite red
    one_min_green_closes = (
        [100.5] + [100.0 - (100.0 - 97.0) * (i + 1) / 14 for i in range(14)] + later
    )
    one_min_green = _session_bars(
        opens=[100.0] + one_min_green_closes[:-1],
        highs=[c + 0.3 for c in one_min_green_closes],
        lows=[c - 0.3 for c in one_min_green_closes],
        closes=one_min_green_closes,
        volumes=[2000.0] * n,
    )

    dip_red = detect_open_dip(red, ticker="RED", prior_close=prior, dip_pct=2.0)
    dip_green = detect_open_dip(green, ticker="GRN", prior_close=prior, dip_pct=2.0)
    dip_1m = detect_open_dip(one_min_green, ticker="ONE", prior_close=prior, dip_pct=2.0)
    assert dip_red.qualifies and dip_green.qualifies and dip_1m.qualifies
    score_red = assess_bounce(red, dip_red)
    score_green = assess_bounce(green, dip_green)
    score_1m = assess_bounce(one_min_green, dip_1m)
    assert score_red.first_bar_green is False
    assert score_green.first_bar_green is True
    assert score_1m.first_bar_green is False  # 15m composite is red despite green first minute
    assert score_green.bounce_score < score_red.bounce_score
    assert any("15m_green_penalty=" in r for r in score_green.reasons)


def test_scan_ranks_candidates(monkeypatch):
    reclaim_closes = [97.0] * 20 + [100.2] * 20
    reclaim = _session_bars(
        opens=[99.0] + reclaim_closes[:-1],
        highs=[c + 0.2 for c in reclaim_closes],
        lows=[96.5] + [c - 0.2 for c in reclaim_closes[1:]],
        closes=reclaim_closes,
        volumes=[1500.0] * 40,
    )
    crash_closes = [100 - i * 0.5 for i in range(30)]
    crash = _session_bars(
        opens=crash_closes,
        highs=[c + 0.1 for c in crash_closes],
        lows=[c - 0.3 for c in crash_closes],
        closes=crash_closes,
    )

    def fake_load(ticker, **kwargs):
        return reclaim if ticker == "GOOD" else crash

    monkeypatch.setattr(
        "tradingai.intraday.scan.load_or_fetch_session_bars",
        fake_load,
    )
    monkeypatch.setattr(
        "tradingai.intraday.scan.prior_daily_close",
        lambda ticker, **kwargs: 100.0,
    )
    monkeypatch.setattr(
        "tradingai.intraday.scan.bullish_sector_watchlist",
        lambda **kwargs: ["GOOD", "BAD"],
    )
    payload = scan_open_dip_bounces(dip_pct=2.0, window_minutes=60, prefer_db=True)
    tickers_c = [row["ticker"] for row in payload["candidates"]]
    tickers_p = [row["ticker"] for row in payload["plunging"]]
    assert "GOOD" in tickers_c or "GOOD" in tickers_p
    assert payload["universe_size"] == 2
    assert "disclaimer" in payload
