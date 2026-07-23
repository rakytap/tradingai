from __future__ import annotations

from datetime import date

import pandas as pd

from tradingai.analysis.ml.config import MIN_HISTORY_ROWS, get_ml_benchmark, get_ml_include_regime
from tradingai.analysis.ml.features import build_indicator_frame, get_feature_columns
from tradingai.analysis.ml.labels import labeled_rows
from tradingai.analysis.ml.macro_history import load_macro_histories
from tradingai.analysis.ml.ohlcv import load_ticker_ohlcv
from tradingai.analysis.ml.regime_features import REGIME_FEATURE_COLUMNS
from tradingai.regime.analysis.historical import build_regime_feature_frame

def _join_regime_features(frame: pd.DataFrame, regime_frame: pd.DataFrame) -> pd.DataFrame:
    if regime_frame.empty:
        return frame

    merged = frame.copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    regime = regime_frame.copy()
    regime["date"] = pd.to_datetime(regime["date"]).dt.date
    merged = merged.merge(regime, on="date", how="left")
    merged = merged.dropna(subset=REGIME_FEATURE_COLUMNS)
    return merged


def build_ticker_dataset(
    ticker: str,
    *,
    horizon_days: int,
    benchmark_df: pd.DataFrame,
    benchmark_symbol: str | None = None,
    min_rows: int = MIN_HISTORY_ROWS,
    database_url: str | None = None,
    regime_frame: pd.DataFrame | None = None,
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

    if get_ml_include_regime() and regime_frame is not None:
        features = _join_regime_features(features, regime_frame)
        if features.empty:
            return pd.DataFrame()

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

    regime_frame: pd.DataFrame | None = None
    if get_ml_include_regime():
        macro_histories = load_macro_histories(database_url=database_url)
        date_index = pd.DatetimeIndex(benchmark_df.index)
        regime_frame = build_regime_feature_frame(date_index, macro_histories)
        if regime_frame.empty:
            raise ValueError("Unable to build regime feature frame from macro histories.")

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
            regime_frame=regime_frame,
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
    X = frame[get_feature_columns()].copy()
    y = frame["direction_up"].astype(int)
    return X, y
