from __future__ import annotations

from datetime import date

import pandas as pd

from tradingai.analysis.ml.config import MIN_HISTORY_ROWS, get_ml_benchmark
from tradingai.analysis.ml.features import FEATURE_COLUMNS, build_indicator_frame
from tradingai.analysis.ml.labels import labeled_rows
from tradingai.data.connection import get_connection


def load_ticker_ohlcv(
    ticker: str,
    *,
    database_url: str | None = None,
) -> pd.DataFrame:
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM daily_prices
            WHERE ticker = %s
            ORDER BY date ASC
            """,
            (ticker.upper(),),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    dates = [row[0] for row in rows]
    data = {
        "open": [float(row[1]) for row in rows],
        "high": [float(row[2]) for row in rows],
        "low": [float(row[3]) for row in rows],
        "close": [float(row[4]) for row in rows],
        "volume": [int(row[5]) for row in rows],
    }
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame(data, index=index)


def build_ticker_dataset(
    ticker: str,
    *,
    horizon_days: int,
    benchmark_df: pd.DataFrame,
    benchmark_symbol: str | None = None,
    min_rows: int = MIN_HISTORY_ROWS,
    database_url: str | None = None,
) -> pd.DataFrame:
    ohlcv = load_ticker_ohlcv(ticker, database_url=database_url)
    if len(ohlcv) < min_rows:
        return pd.DataFrame()

    benchmark = benchmark_symbol or get_ml_benchmark()
    features = build_indicator_frame(
        ohlcv,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark,
    )
    features["ticker"] = ticker.upper()
    labeled = labeled_rows(features, horizon_days=horizon_days)
    if labeled.empty:
        return pd.DataFrame()
    labeled["date"] = pd.to_datetime(labeled["date"]).dt.date
    return labeled


def build_training_dataset(
    tickers: list[str],
    *,
    horizon_days: int,
    benchmark_ticker: str | None = None,
    min_rows: int = MIN_HISTORY_ROWS,
    database_url: str | None = None,
) -> pd.DataFrame:
    benchmark = benchmark_ticker or get_ml_benchmark()
    benchmark_df = load_ticker_ohlcv(benchmark, database_url=database_url)
    if benchmark_df.empty:
        raise ValueError(
            f"Benchmark ticker '{benchmark}' has no price history. Run tradingai-sync-prices first."
        )

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        normalized = ticker.upper().strip()
        if normalized == benchmark:
            continue
        frame = build_ticker_dataset(
            normalized,
            horizon_days=horizon_days,
            benchmark_df=benchmark_df,
            benchmark_symbol=benchmark,
            min_rows=min_rows,
            database_url=database_url,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def chronological_split(
    dataset: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset.empty:
        return dataset.copy(), dataset.copy()

    ordered_dates = sorted(dataset["date"].unique())
    if len(ordered_dates) < 2:
        return dataset.copy(), dataset.iloc[0:0].copy()

    split_index = max(1, int(len(ordered_dates) * train_fraction))
    if split_index >= len(ordered_dates):
        split_index = len(ordered_dates) - 1

    train_cutoff: date = ordered_dates[split_index - 1]
    train = dataset[dataset["date"] <= train_cutoff].copy()
    test = dataset[dataset["date"] > train_cutoff].copy()
    return train, test


def split_features_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = frame[FEATURE_COLUMNS].copy()
    y = frame["direction_up"].astype(int)
    return X, y
