from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingai.intraday.bounce import assess_bounce
from tradingai.intraday.candles import (
    assess_candle_patterns,
    candle_features,
    is_bullish_engulfing,
    is_hammer,
    is_morning_star,
    resample_ohlcv,
)
from tradingai.intraday.open_dip import detect_open_dip

ET = ZoneInfo("America/New_York")


def test_candle_features_geometry():
    feat = candle_features(10.0, 12.0, 9.0, 11.0)
    assert feat.body == pytest.approx(1.0)
    assert feat.range == pytest.approx(3.0)
    assert feat.lower_wick == pytest.approx(1.0)
    assert feat.upper_wick == pytest.approx(1.0)
    assert feat.is_green is True


def test_is_hammer_true_and_false():
    hammer = candle_features(10.0, 10.2, 8.0, 10.1)  # long lower wick, small body high
    assert is_hammer(hammer) is True
    no_hammer = candle_features(10.0, 11.0, 9.8, 10.5)  # balanced
    assert is_hammer(no_hammer) is False


def test_bullish_engulfing():
    prev = candle_features(10.5, 10.6, 9.8, 9.9)  # red
    cur = candle_features(9.85, 10.8, 9.7, 10.7)  # green engulfing
    assert is_bullish_engulfing(prev, cur) is True
    assert is_bullish_engulfing(cur, prev) is False


def test_morning_star():
    first = candle_features(10.0, 10.1, 9.0, 9.1)
    mid = candle_features(9.05, 9.15, 8.7, 8.9)
    third = candle_features(9.0, 10.2, 8.95, 10.0)
    assert is_morning_star(first, mid, third) is True


def test_resample_ohlcv_5m():
    start = datetime(2026, 10, 14, 9, 30, tzinfo=ET)
    index = pd.date_range(start, periods=10, freq="1min", tz=ET)
    bars = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(10)],
            "high": [100.5 + i * 0.1 for i in range(10)],
            "low": [99.5 + i * 0.1 for i in range(10)],
            "close": [100.2 + i * 0.1 for i in range(10)],
            "volume": [100.0] * 10,
        },
        index=index,
    )
    out = resample_ohlcv(bars, rule="5min")
    assert len(out) == 2
    assert float(out.iloc[0]["open"]) == pytest.approx(100.0)
    assert float(out.iloc[0]["close"]) == pytest.approx(100.6)
    assert float(out.iloc[0]["volume"]) == pytest.approx(500.0)


def _session_bars(
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    start = datetime(2026, 10, 14, 9, 30, tzinfo=ET)
    index = pd.date_range(start, periods=len(closes), freq="1min", tz=ET)
    volumes = volumes or [2000.0] * len(closes)
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


def test_assess_candle_patterns_detects_trough_hammer():
    # Build 1m bars that form a hammer on the 5m bucket at trough.
    # 09:30–09:34: selloff; 09:35–09:39: hammer-like recovery at lows.
    opens, highs, lows, closes = [], [], [], []
    for i in range(5):
        # Red selloff into 97
        opens.append(100.0 - i * 0.5)
        highs.append(opens[-1] + 0.1)
        lows.append(97.5 - i * 0.1)
        closes.append(lows[-1] + 0.05)
    # Next 5m: long lower wick hammer ending near high of range
    for i in range(5):
        opens.append(97.2)
        highs.append(97.8)
        lows.append(96.0)  # long lower wick vs body
        closes.append(97.6)
    for i in range(20):
        opens.append(97.6)
        highs.append(98.0)
        lows.append(97.4)
        closes.append(97.7)

    bars = _session_bars(opens=opens, highs=highs, lows=lows, closes=closes)
    dip = detect_open_dip(bars, ticker="HAM", prior_close=100.0, dip_pct=2.0, window_minutes=60)
    assert dip.qualifies
    assert dip.trough_time is not None
    patterns = assess_candle_patterns(bars, trough_time=dip.trough_time)
    assert patterns.trough_hammer is True
    assert patterns.score_delta >= 0.3


def test_bounce_score_increases_with_trough_hammer():
    prior = 100.0
    # Mild bounce (not full reclaim) so scores stay below the +1 clip and hammer can lift rank.
    # Sell into ~97, then grind to ~98.2 (above open path weakly / near VWAP).
    closes = [100.0 - (100.0 - 97.0) * (i + 1) / 12 for i in range(12)] + [97.2] * 5 + [98.2] * 13
    n = len(closes)
    base = _session_bars(
        opens=[100.0] + closes[:-1],
        highs=[c + 0.2 for c in closes],
        lows=[c - 0.2 for c in closes],
        closes=closes,
        volumes=[2000.0] * n,
    )
    hammer_lows = list(base["low"])
    hammer_highs = list(base["high"])
    hammer_closes = list(base["close"])
    hammer_opens = list(base["open"])
    # Force a 5m hammer around minutes 5–9 (trough + long lower wick).
    for i in range(5, 10):
        hammer_lows[i] = 95.0
        hammer_opens[i] = 96.8
        hammer_closes[i] = 97.2
        hammer_highs[i] = 97.35
    hammer_bars = _session_bars(
        opens=hammer_opens,
        highs=hammer_highs,
        lows=hammer_lows,
        closes=hammer_closes,
        volumes=[2000.0] * n,
    )

    dip_base = detect_open_dip(base, ticker="BASE", prior_close=prior, dip_pct=2.0)
    dip_hammer = detect_open_dip(hammer_bars, ticker="HAM", prior_close=prior, dip_pct=2.0)
    assert dip_base.qualifies and dip_hammer.qualifies
    score_base = assess_bounce(base, dip_base).bounce_score
    bounce_h = assess_bounce(hammer_bars, dip_hammer)
    assert bounce_h.trough_hammer is True
    assert bounce_h.bounce_score > score_base
    assert score_base < 1.0  # room for the hammer additive to matter
