from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingai.intraday.config import (
    DEFAULT_FIRST_BAR_GREEN_PENALTY,
    DEFAULT_FIRST_BAR_MINUTES,
    DEFAULT_LOWER_LOW_BARS,
    DEFAULT_MOMENTUM_MINUTES,
)
from tradingai.intraday.open_dip import OpenDipSignal, is_making_lower_lows, latest_vwap


Status = str  # plunging | stabilizing | reclaiming


@dataclass(frozen=True)
class BounceAssessment:
    ticker: str
    status: Status
    bounce_score: float
    last_price: float | None
    vwap: float | None
    reclaim_vwap: bool
    reclaim_open: bool
    first_bar_green: bool
    momentum_15m_pct: float | None
    reasons: tuple[str, ...]


def opening_window(bars: pd.DataFrame, *, minutes: int = DEFAULT_FIRST_BAR_MINUTES) -> pd.DataFrame:
    """First N minutes of RTH as a slice of 1m bars."""
    if bars is None or bars.empty:
        return bars
    n = max(1, int(minutes))
    if isinstance(bars.index, pd.DatetimeIndex):
        start = bars.index[0]
        end = start + pd.Timedelta(minutes=n)
        window = bars.loc[(bars.index >= start) & (bars.index < end)]
        if not window.empty:
            return window
    return bars.iloc[:n]


def first_bar_is_green(
    bars: pd.DataFrame,
    *,
    minutes: int = DEFAULT_FIRST_BAR_MINUTES,
) -> bool:
    """True when the opening N-minute composite bar is green (close > open).

    Composite open = first 1m open; composite close = last 1m close in the window.
    Default window is 15 minutes (not the first 1m candle alone).
    """
    if bars is None or bars.empty:
        return False
    if "open" not in bars.columns or "close" not in bars.columns:
        return False
    window = opening_window(bars, minutes=minutes)
    if window.empty:
        return False
    first_open = float(window["open"].iloc[0])
    first_close = float(window["close"].iloc[-1])
    return first_close > first_open


def assess_bounce(
    bars: pd.DataFrame,
    dip: OpenDipSignal,
    *,
    momentum_minutes: int = DEFAULT_MOMENTUM_MINUTES,
    lower_low_bars: int = DEFAULT_LOWER_LOW_BARS,
    first_bar_minutes: int = DEFAULT_FIRST_BAR_MINUTES,
    first_bar_green_penalty: float = DEFAULT_FIRST_BAR_GREEN_PENALTY,
) -> BounceAssessment:
    """Score recovery after an open dip (deterministic blend in [-1, 1])."""
    ticker = dip.ticker
    if bars is None or bars.empty:
        return BounceAssessment(
            ticker=ticker,
            status="plunging",
            bounce_score=-1.0,
            last_price=None,
            vwap=None,
            reclaim_vwap=False,
            reclaim_open=False,
            first_bar_green=False,
            momentum_15m_pct=None,
            reasons=("no_bars",),
        )

    last_price = float(bars["close"].iloc[-1])
    vwap = latest_vwap(bars)
    session_open = dip.session_open
    reclaim_vwap = bool(vwap is not None and last_price >= vwap)
    reclaim_open = bool(session_open is not None and last_price >= session_open)
    green_open = first_bar_is_green(bars, minutes=first_bar_minutes)

    n = max(1, int(momentum_minutes))
    if len(bars) > n:
        past = float(bars["close"].iloc[-(n + 1)])
        momentum = (last_price / past - 1.0) * 100.0 if past > 0 else 0.0
    else:
        momentum = 0.0
    momentum_positive = momentum > 0

    free_fall = is_making_lower_lows(bars, lookback=lower_low_bars)
    reasons: list[str] = []

    # Score components in [-1, 1]
    dip_component = max(-1.0, min(0.0, dip.dip_pct / 10.0))  # deeper dip → more negative base
    vwap_component = 0.45 if reclaim_vwap else -0.25
    open_component = 0.35 if reclaim_open else -0.15
    mom_component = 0.35 if momentum_positive else -0.35
    if free_fall:
        mom_component = -0.6
        reasons.append(f"lower_lows_last_{lower_low_bars}")

    score = 0.15 * dip_component + vwap_component + open_component + mom_component
    # Opening green then selling off is a weaker bounce setup — penalize.
    if green_open:
        penalty = max(0.0, float(first_bar_green_penalty))
        score -= penalty
        reasons.append(
            f"first_{first_bar_minutes}m_green_penalty=-{penalty:.2f}"
        )

    score = max(-1.0, min(1.0, score))
    # Shift so reclaiming setups land positive (still after green-open penalty).
    if reclaim_vwap and reclaim_open and momentum_positive and not free_fall and not green_open:
        score = max(score, 0.35)

    if free_fall or (not reclaim_vwap and not reclaim_open and not momentum_positive):
        status: Status = "plunging"
    elif reclaim_vwap or reclaim_open:
        status = "reclaiming"
    else:
        status = "stabilizing"

    if reclaim_vwap:
        reasons.append("reclaim_vwap")
    if reclaim_open:
        reasons.append("reclaim_session_open")
    if momentum_positive:
        reasons.append(f"momentum_{momentum_minutes}m={momentum:.2f}%")
    else:
        reasons.append(f"momentum_{momentum_minutes}m={momentum:.2f}%")

    return BounceAssessment(
        ticker=ticker,
        status=status,
        bounce_score=round(float(score), 4),
        last_price=round(last_price, 6),
        vwap=round(vwap, 6) if vwap is not None else None,
        reclaim_vwap=reclaim_vwap,
        reclaim_open=reclaim_open,
        first_bar_green=green_open,
        momentum_15m_pct=round(momentum, 4),
        reasons=tuple(reasons),
    )
