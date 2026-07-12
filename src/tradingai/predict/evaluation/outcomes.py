from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from tradingai.predict.storage.repository import (
    get_predictions_pending_outcomes,
    save_outcome,
)


def _normalize_close(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("No price data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0]
        return close
    return df["Close"]


def _fetch_history(ticker: str, start: str, end: str) -> pd.Series:
    data = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    return _normalize_close(data)


def _next_trading_day_after(series: pd.Series, date_str: str, offset: int) -> float | None:
    if series.empty:
        return None
    index = pd.to_datetime(series.index).tz_localize(None)
    series = pd.Series(series.values, index=index)
    target = pd.Timestamp(date_str)
    future = series[series.index > target]
    if future.empty:
        return None
    start_close = future.iloc[0]
    if offset >= len(future):
        return None
    end_close = future.iloc[offset]
    return float((end_close / start_close - 1.0) * 100.0)


def evaluate_pending_outcomes(
    horizon_days: int,
    *,
    db_path: str | None = None,
    as_of: datetime | None = None,
) -> dict:
    as_of = as_of or datetime.utcnow()
    pending = get_predictions_pending_outcomes(horizon_days, db_path=db_path)
    evaluated = 0
    skipped = 0
    errors: list[str] = []

    for row in pending:
        prediction_date = datetime.fromisoformat(row["prediction_date"])
        if prediction_date + timedelta(days=horizon_days + 5) > as_of:
            skipped += 1
            continue
        ticker = row["ticker"]
        start = (prediction_date - timedelta(days=3)).date().isoformat()
        end = (prediction_date + timedelta(days=horizon_days + 30)).date().isoformat()
        try:
            history = _fetch_history(ticker, start, end)
            actual_return = _next_trading_day_after(history, row["prediction_date"], horizon_days)
            if actual_return is None:
                skipped += 1
                continue
            save_outcome(
                row["id"],
                horizon_days,
                actual_return,
                db_path=db_path,
            )
            evaluated += 1
        except Exception as exc:
            errors.append(f"{ticker}:{row['prediction_date']}:{exc}")
            skipped += 1

    return {
        "horizon_days": horizon_days,
        "evaluated": evaluated,
        "skipped": skipped,
        "errors": errors[:10],
    }
