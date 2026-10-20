from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingai.intraday.bars import rth_open_for_day, session_vwap
from tradingai.intraday.config import (
    DEFAULT_DIP_PCT,
    DEFAULT_DIP_WINDOW_MINUTES,
    DEFAULT_LOWER_LOW_BARS,
)


@dataclass(frozen=True)
class OpenDipSignal:
    ticker: str
    qualifies: bool
    dip_pct: float
    trough_price: float | None
    prior_close: float | None
    session_open: float | None
    window_minutes: int
    reasons: tuple[str, ...]
    trough_time: pd.Timestamp | None = None
    trough_hammer: bool = False
    bullish_engulfing: bool = False
    candle_reasons: tuple[str, ...] = ()


def detect_open_dip(
    bars: pd.DataFrame,
    *,
    ticker: str,
    prior_close: float | None,
    dip_pct: float = DEFAULT_DIP_PCT,
    window_minutes: int = DEFAULT_DIP_WINDOW_MINUTES,
) -> OpenDipSignal:
    """Detect a morning plunge vs prior close within the first ``window_minutes``."""
    reasons: list[str] = []
    if bars is None or bars.empty:
        return OpenDipSignal(
            ticker=ticker.upper(),
            qualifies=False,
            dip_pct=0.0,
            trough_price=None,
            prior_close=prior_close,
            session_open=None,
            window_minutes=window_minutes,
            reasons=("no_intraday_bars",),
        )
    if prior_close is None or prior_close <= 0:
        return OpenDipSignal(
            ticker=ticker.upper(),
            qualifies=False,
            dip_pct=0.0,
            trough_price=None,
            prior_close=prior_close,
            session_open=float(bars["open"].iloc[0]) if "open" in bars.columns else None,
            window_minutes=window_minutes,
            reasons=("missing_prior_close",),
        )

    day = bars.index[0].date()
    open_ts = rth_open_for_day(day)
    window_end = open_ts + pd.Timedelta(minutes=max(1, int(window_minutes)))
    window = bars.loc[(bars.index >= open_ts) & (bars.index < window_end)]
    if window.empty:
        window = bars.iloc[: max(1, int(window_minutes))]

    trough_idx = window["low"].astype(float).idxmin()
    trough = float(window.loc[trough_idx, "low"])
    trough_time = pd.Timestamp(trough_idx)
    session_open = float(bars["open"].iloc[0])
    dip = (trough / prior_close - 1.0) * 100.0
    threshold = -abs(float(dip_pct))
    qualifies = dip <= threshold
    if qualifies:
        reasons.append(f"trough {dip:.2f}% vs prior close (threshold {threshold:.2f}%)")
    else:
        reasons.append(f"dip {dip:.2f}% shallower than {threshold:.2f}%")

    # Informational selloff quality (not a gate).
    red = (window["close"].astype(float) < window["open"].astype(float)).sum()
    red_frac = float(red) / float(len(window)) if len(window) else 0.0
    if qualifies:
        reasons.append(f"red_body_fraction={red_frac:.2f}")

    return OpenDipSignal(
        ticker=ticker.upper(),
        qualifies=qualifies,
        dip_pct=round(dip, 4),
        trough_price=round(trough, 6),
        prior_close=round(float(prior_close), 6),
        session_open=round(session_open, 6),
        window_minutes=int(window_minutes),
        reasons=tuple(reasons),
        trough_time=trough_time if qualifies else None,
    )


def is_making_lower_lows(
    bars: pd.DataFrame,
    *,
    lookback: int = DEFAULT_LOWER_LOW_BARS,
) -> bool:
    """True when the last ``lookback`` lows are strictly descending."""
    if bars is None or bars.empty or "low" not in bars.columns:
        return False
    n = max(2, int(lookback))
    if len(bars) < n:
        return False
    lows = bars["low"].astype(float).iloc[-n:].to_numpy()
    return all(lows[i] < lows[i - 1] for i in range(1, len(lows)))


def latest_vwap(bars: pd.DataFrame) -> float | None:
    series = session_vwap(bars)
    if series.empty:
        return None
    value = series.dropna()
    if value.empty:
        return None
    return float(value.iloc[-1])
