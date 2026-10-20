"""Deterministic Japanese candle geometry and patterns for open-dip bounce."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingai.intraday.config import (
    DEFAULT_CANDLE_HAMMER_LOWER_WICK_RATIO,
    DEFAULT_CANDLE_HAMMER_MAX_UPPER_WICK_RATIO,
    DEFAULT_CANDLE_MIN_REL_VOLUME,
    DEFAULT_CANDLE_SCORE_BEARISH_ENGULF,
    DEFAULT_CANDLE_SCORE_BULLISH_ENGULF,
    DEFAULT_CANDLE_SCORE_HAMMER,
)


@dataclass(frozen=True)
class CandleFeatures:
    open: float
    high: float
    low: float
    close: float
    body: float
    range: float
    body_pct: float
    upper_wick: float
    lower_wick: float
    close_loc: float
    is_green: bool
    is_red: bool


@dataclass(frozen=True)
class CandlePatternAssessment:
    trough_hammer: bool
    bullish_engulfing: bool
    bearish_engulfing: bool
    morning_star: bool
    opening_marubozu_green: bool
    opening_long_upper_wick: bool
    red_body_fraction: float | None
    score_delta: float
    reasons: tuple[str, ...]


def candle_features(open_: float, high: float, low: float, close: float) -> CandleFeatures:
    o, h, l, c = float(open_), float(high), float(low), float(close)
    body = abs(c - o)
    rng = max(h - l, 0.0)
    if rng <= 0:
        body_pct = 0.0
        upper = 0.0
        lower = 0.0
        close_loc = 0.5
    else:
        body_pct = body / rng
        upper = h - max(o, c)
        lower = min(o, c) - l
        close_loc = (c - l) / rng
    return CandleFeatures(
        open=o,
        high=h,
        low=l,
        close=c,
        body=body,
        range=rng,
        body_pct=body_pct,
        upper_wick=max(0.0, upper),
        lower_wick=max(0.0, lower),
        close_loc=close_loc,
        is_green=c > o,
        is_red=c < o,
    )


def row_features(row: pd.Series) -> CandleFeatures:
    return candle_features(
        float(row["open"]),
        float(row["high"]),
        float(row["low"]),
        float(row["close"]),
    )


def is_hammer(
    feat: CandleFeatures,
    *,
    lower_wick_ratio: float = DEFAULT_CANDLE_HAMMER_LOWER_WICK_RATIO,
    max_upper_wick_ratio: float = DEFAULT_CANDLE_HAMMER_MAX_UPPER_WICK_RATIO,
) -> bool:
    """Hammer / long lower wick: rejection of lows, body near top of range."""
    if feat.range <= 0:
        return False
    body = max(feat.body, feat.range * 0.05)
    upper_ok = (
        feat.upper_wick <= max_upper_wick_ratio * body
        or feat.upper_wick / feat.range <= 0.25
    )
    return (
        feat.lower_wick >= lower_wick_ratio * body
        and upper_ok
        and feat.close_loc >= 2.0 / 3.0
    )


def is_bullish_engulfing(prev: CandleFeatures, cur: CandleFeatures) -> bool:
    if not prev.is_red or not cur.is_green:
        return False
    prev_top = max(prev.open, prev.close)
    prev_bot = min(prev.open, prev.close)
    cur_top = max(cur.open, cur.close)
    cur_bot = min(cur.open, cur.close)
    return cur_bot <= prev_bot and cur_top >= prev_top and cur.body > prev.body


def is_bearish_engulfing(prev: CandleFeatures, cur: CandleFeatures) -> bool:
    if not prev.is_green or not cur.is_red:
        return False
    prev_top = max(prev.open, prev.close)
    prev_bot = min(prev.open, prev.close)
    cur_top = max(cur.open, cur.close)
    cur_bot = min(cur.open, cur.close)
    return cur_bot <= prev_bot and cur_top >= prev_top and cur.body > prev.body


def is_morning_star(first: CandleFeatures, mid: CandleFeatures, third: CandleFeatures) -> bool:
    """Classic morning star on three bars (typically 5m)."""
    if not first.is_red or not third.is_green:
        return False
    if first.body <= 0 or third.body <= 0:
        return False
    # Mid is small body near/below first close zone.
    if mid.body_pct > 0.35:
        return False
    mid_max = max(mid.open, mid.close)
    if mid_max > min(first.open, first.close):
        return False
    # Third closes well into first body.
    first_mid = (first.open + first.close) / 2.0
    return third.close >= first_mid


def is_green_marubozu(feat: CandleFeatures, *, min_body_pct: float = 0.75) -> bool:
    return feat.is_green and feat.body_pct >= min_body_pct and feat.lower_wick <= feat.range * 0.08


def has_long_upper_wick(feat: CandleFeatures, *, min_ratio: float = 1.5) -> bool:
    if feat.body <= 0:
        return feat.upper_wick >= feat.range * 0.4
    return feat.upper_wick >= min_ratio * feat.body


def resample_ohlcv(bars: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """Aggregate 1m OHLCV to a coarser session timeframe."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    work = bars.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("resample_ohlcv requires a DatetimeIndex")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in work.columns:
        agg["volume"] = "sum"
    out = work.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open", "close"])
    return out


def _passes_volume_gate(
    bar_volume: float,
    session_volumes: pd.Series,
    *,
    min_rel_volume: float = DEFAULT_CANDLE_MIN_REL_VOLUME,
) -> bool:
    if session_volumes is None or session_volumes.empty:
        return True
    median = float(session_volumes.median())
    if median <= 0:
        return True
    return float(bar_volume) >= min_rel_volume * median


def opening_composite_features(
    bars: pd.DataFrame,
    *,
    minutes: int = 15,
) -> CandleFeatures | None:
    if bars is None or bars.empty:
        return None
    n = max(1, int(minutes))
    start = bars.index[0]
    end = start + pd.Timedelta(minutes=n)
    window = bars.loc[(bars.index >= start) & (bars.index < end)]
    if window.empty:
        window = bars.iloc[:n]
    if window.empty:
        return None
    return candle_features(
        float(window["open"].iloc[0]),
        float(window["high"].max()),
        float(window["low"].min()),
        float(window["close"].iloc[-1]),
    )


def red_body_fraction(bars: pd.DataFrame) -> float | None:
    if bars is None or bars.empty or "open" not in bars.columns:
        return None
    red = (bars["close"].astype(float) < bars["open"].astype(float)).sum()
    return float(red) / float(len(bars))


def assess_candle_patterns(
    bars_1m: pd.DataFrame,
    *,
    trough_time: pd.Timestamp | None,
    last_price: float | None = None,
    vwap: float | None = None,
    first_bar_minutes: int = 15,
    hammer_weight: float = DEFAULT_CANDLE_SCORE_HAMMER,
    bullish_engulf_weight: float = DEFAULT_CANDLE_SCORE_BULLISH_ENGULF,
    bearish_engulf_weight: float = DEFAULT_CANDLE_SCORE_BEARISH_ENGULF,
    min_rel_volume: float = DEFAULT_CANDLE_MIN_REL_VOLUME,
) -> CandlePatternAssessment:
    """Score candle confirmation around the dip trough (5m patterns)."""
    reasons: list[str] = []
    trough_hammer = False
    bullish_engulf = False
    bearish_engulf = False
    morning_star = False
    opening_marubozu = False
    opening_long_upper = False
    score = 0.0

    red_frac = red_body_fraction(bars_1m)
    if red_frac is not None:
        reasons.append(f"red_body_fraction={red_frac:.2f}")

    open_feat = opening_composite_features(bars_1m, minutes=first_bar_minutes)
    if open_feat is not None:
        if is_green_marubozu(open_feat):
            opening_marubozu = True
            reasons.append("opening_15m_green_marubozu")
        if has_long_upper_wick(open_feat):
            opening_long_upper = True
            reasons.append("opening_15m_long_upper_wick")

    if bars_1m is None or bars_1m.empty or trough_time is None:
        return CandlePatternAssessment(
            trough_hammer=False,
            bullish_engulfing=False,
            bearish_engulfing=False,
            morning_star=False,
            opening_marubozu_green=opening_marubozu,
            opening_long_upper_wick=opening_long_upper,
            red_body_fraction=red_frac,
            score_delta=0.0,
            reasons=tuple(reasons),
        )

    bars_5m = resample_ohlcv(bars_1m, rule="5min")
    if bars_5m.empty:
        return CandlePatternAssessment(
            trough_hammer=False,
            bullish_engulfing=False,
            bearish_engulfing=False,
            morning_star=False,
            opening_marubozu_green=opening_marubozu,
            opening_long_upper_wick=opening_long_upper,
            red_body_fraction=red_frac,
            score_delta=0.0,
            reasons=tuple(reasons),
        )

    vol_series = bars_5m["volume"] if "volume" in bars_5m.columns else pd.Series(dtype=float)
    # Index of 5m bar containing trough (label=left).
    pos = bars_5m.index.searchsorted(trough_time, side="right") - 1
    pos = max(0, min(pos, len(bars_5m) - 1))

    # Hammer on trough bar or ±1 neighbour.
    for idx in {pos, max(0, pos - 1), min(len(bars_5m) - 1, pos + 1)}:
        row = bars_5m.iloc[idx]
        feat = row_features(row)
        vol = float(row["volume"]) if "volume" in bars_5m.columns else 0.0
        if not _passes_volume_gate(vol, vol_series, min_rel_volume=min_rel_volume):
            continue
        if is_hammer(feat):
            trough_hammer = True
            reasons.append("trough_hammer_5m")
            break

    # Bullish engulfing only in the first couple of 5m bars after the trough.
    for i in range(pos, min(pos + 2, len(bars_5m) - 1)):
        prev = row_features(bars_5m.iloc[i])
        cur = row_features(bars_5m.iloc[i + 1])
        vol = float(bars_5m.iloc[i + 1]["volume"]) if "volume" in bars_5m.columns else 0.0
        if not _passes_volume_gate(vol, vol_series, min_rel_volume=min_rel_volume):
            continue
        if is_bullish_engulfing(prev, cur):
            bullish_engulf = True
            reasons.append("bullish_engulfing_5m")
            break

    # Morning star spanning trough region.
    if pos >= 1 and pos + 1 < len(bars_5m):
        a = row_features(bars_5m.iloc[pos - 1])
        b = row_features(bars_5m.iloc[pos])
        c = row_features(bars_5m.iloc[pos + 1])
        vol = float(bars_5m.iloc[pos]["volume"]) if "volume" in bars_5m.columns else 0.0
        if _passes_volume_gate(vol, vol_series, min_rel_volume=min_rel_volume) and is_morning_star(
            a, b, c
        ):
            morning_star = True
            reasons.append("morning_star_5m")

    # Bearish engulfing reclaim failure near the end (still below VWAP).
    if len(bars_5m) >= 2:
        prev = row_features(bars_5m.iloc[-2])
        cur = row_features(bars_5m.iloc[-1])
        vol = float(bars_5m.iloc[-1]["volume"]) if "volume" in bars_5m.columns else 0.0
        below_vwap = vwap is not None and last_price is not None and last_price < vwap
        if (
            below_vwap
            and _passes_volume_gate(vol, vol_series, min_rel_volume=min_rel_volume)
            and is_bearish_engulfing(prev, cur)
        ):
            bearish_engulf = True
            reasons.append("bearish_engulfing_reclaim_fail_5m")

    if trough_hammer:
        score += float(hammer_weight)
    if bullish_engulf or morning_star:
        score += float(bullish_engulf_weight)
    if bearish_engulf:
        score -= float(bearish_engulf_weight)
    if opening_marubozu:
        score -= 0.1  # extra beyond the generic green-open penalty

    return CandlePatternAssessment(
        trough_hammer=trough_hammer,
        bullish_engulfing=bullish_engulf,
        bearish_engulfing=bearish_engulf,
        morning_star=morning_star,
        opening_marubozu_green=opening_marubozu,
        opening_long_upper_wick=opening_long_upper,
        red_body_fraction=round(red_frac, 4) if red_frac is not None else None,
        score_delta=round(score, 4),
        reasons=tuple(reasons),
    )
