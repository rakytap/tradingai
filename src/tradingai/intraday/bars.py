from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd

from tradingai.intraday.config import (
    ET,
    RTH_OPEN_HOUR,
    RTH_OPEN_MINUTE,
    get_bar_interval,
    get_bar_period,
)


def rth_open_for_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, RTH_OPEN_HOUR, RTH_OPEN_MINUTE, tzinfo=ET)


def normalize_intraday_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV columns and America/New_York index."""
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("Intraday bars require a DatetimeIndex.")
    if result.index.tz is None:
        result.index = result.index.tz_localize(ET)
    else:
        result.index = result.index.tz_convert(ET)
    result = result.sort_index()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in result.columns]
    return result[cols].astype(float)


def filter_rth_session(frame: pd.DataFrame, *, session_day: date | None = None) -> pd.DataFrame:
    """Keep regular-session bars (09:30–16:00 ET) for one calendar day."""
    if frame.empty:
        return frame
    work = normalize_intraday_index(frame)
    day = session_day or work.index[-1].date()
    open_ts = rth_open_for_day(day)
    close_ts = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)
    return work.loc[(work.index >= open_ts) & (work.index < close_ts)].copy()


def bars_to_rows(frame: pd.DataFrame) -> list[dict]:
    work = normalize_intraday_index(frame)
    rows: list[dict] = []
    for ts, row in work.iterrows():
        rows.append(
            {
                "bar_time": ts.to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if "volume" in work.columns else 0.0,
            }
        )
    return rows


def fetch_intraday_ohlcv(
    ticker: str,
    *,
    period: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    from tradingai.providers.registry import get_provider

    provider = get_provider()
    raw = provider.get_ohlcv(
        ticker.upper(),
        period or get_bar_period(),
        interval or get_bar_interval(),
    )
    return normalize_intraday_index(raw)


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP over the provided session bars."""
    if frame.empty or "volume" not in frame.columns:
        return pd.Series(dtype=float)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    vol = frame["volume"].clip(lower=0.0)
    cum_vol = vol.cumsum()
    cum_pv = (typical * vol).cumsum()
    vwap = cum_pv / cum_vol.replace(0.0, pd.NA)
    return vwap.astype(float)


def prior_daily_close(ticker: str, *, before: date | None = None) -> float | None:
    """Latest daily close strictly before ``before`` (default: today ET)."""
    from tradingai.data.repositories.daily_prices import is_finite_number
    from tradingai.data.connection import get_connection

    day = before or datetime.now(ET).date()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT close FROM daily_prices
            WHERE ticker = %s AND date < %s
            ORDER BY date DESC LIMIT 1
            """,
            (ticker.upper(), day),
        ).fetchone()
    if row and is_finite_number(row[0]):
        return float(row[0])
    # Fallback: yfinance 1d
    try:
        from tradingai.providers.registry import get_provider

        hist = get_provider().get_ohlcv(ticker.upper(), "10d", "1d")
        if hist.empty or "close" not in hist.columns:
            return None
        closes = hist["close"].dropna()
        if closes.empty:
            return None
        # Drop same-day partial if present
        idx = pd.to_datetime(closes.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(ET)
        mask = [ts.date() < day for ts in idx]
        filtered = closes.loc[mask]
        if filtered.empty:
            return float(closes.iloc[-1])
        return float(filtered.iloc[-1])
    except Exception:
        return None


def load_or_fetch_session_bars(
    ticker: str,
    *,
    session_day: date | None = None,
    prefer_db: bool = True,
) -> pd.DataFrame:
    day = session_day or datetime.now(ET).date()
    if prefer_db:
        from tradingai.data.repositories.intraday_bars import load_session_bars

        db_bars = load_session_bars(ticker, session_date=datetime.combine(day, time(0, 0), tzinfo=ET))
        if not db_bars.empty:
            return filter_rth_session(db_bars, session_day=day)
    fetched = fetch_intraday_ohlcv(ticker)
    return filter_rth_session(fetched, session_day=day)
