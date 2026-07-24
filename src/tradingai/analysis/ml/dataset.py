from __future__ import annotations

from datetime import date

import pandas as pd

from tradingai.analysis.ml.config import (
    MIN_CS_PEERS,
    MIN_HISTORY_ROWS,
    get_ml_benchmark,
    get_ml_include_regime,
    get_ml_label_mode,
)
from tradingai.analysis.ml.features import build_indicator_frame, get_feature_columns
from tradingai.analysis.ml.labels import labeled_rows, labeled_rows_cross_section
from tradingai.analysis.ml.macro_history import load_macro_histories
from tradingai.analysis.ml.ohlcv import load_ticker_ohlcv
from tradingai.analysis.ml.regime_features import REGIME_FEATURE_COLUMNS
from tradingai.data.repositories.stocks import get_stock
from tradingai.regime.analysis.historical import build_regime_feature_frame
from tradingai.regime.sector_proxies import normalize_sector_label, sector_regime_proxy


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


def _ticker_sector(ticker: str, *, database_url: str | None = None) -> str | None:
    try:
        stock = get_stock(ticker, database_url=database_url)
    except Exception:
        return None
    if not stock:
        return None
    return normalize_sector_label(stock.get("sector")) or stock.get("sector")


def build_ticker_dataset(
    ticker: str,
    *,
    horizon_days: int,
    benchmark_df: pd.DataFrame,
    benchmark_symbol: str | None = None,
    min_rows: int = MIN_HISTORY_ROWS,
    database_url: str | None = None,
    regime_frame: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
    sector: str | None = None,
) -> pd.DataFrame:
    ohlcv = load_ticker_ohlcv(ticker, database_url=database_url)
    if len(ohlcv) < min_rows:
        return pd.DataFrame()

    benchmark = benchmark_symbol or get_ml_benchmark()
    resolved_sector = sector if sector is not None else _ticker_sector(ticker, database_url=database_url)
    proxy_df = sector_df
    if proxy_df is None and resolved_sector:
        proxy = sector_regime_proxy(resolved_sector)
        if proxy.upper() != (benchmark or "").upper():
            proxy_df = load_ticker_ohlcv(proxy, database_url=database_url)
        else:
            proxy_df = benchmark_df

    features = build_indicator_frame(
        ohlcv,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark,
        sector_df=proxy_df if proxy_df is not None and not proxy_df.empty else benchmark_df,
    )
    features["ticker"] = ticker.upper()
    features["sector"] = resolved_sector

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
    label_mode: str | None = None,
    min_cs_peers: int = MIN_CS_PEERS,
) -> pd.DataFrame:
    benchmark = benchmark_ticker or get_ml_benchmark()
    mode = label_mode or get_ml_label_mode()
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

    sector_cache: dict[str, pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        normalized = ticker.upper().strip()
        if normalized == benchmark:
            continue
        sector = _ticker_sector(normalized, database_url=database_url)
        proxy = sector_regime_proxy(sector)
        if proxy not in sector_cache:
            if proxy.upper() == benchmark.upper():
                sector_cache[proxy] = benchmark_df
            else:
                sector_cache[proxy] = load_ticker_ohlcv(proxy, database_url=database_url)
        frame = build_ticker_dataset(
            normalized,
            horizon_days=horizon_days,
            benchmark_df=benchmark_df,
            benchmark_symbol=benchmark,
            min_rows=min_rows,
            database_url=database_url,
            regime_frame=regime_frame,
            sector_df=sector_cache[proxy],
            sector=sector,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    if mode == "cross_section":
        return labeled_rows_cross_section(pooled, min_peers=min_cs_peers)
    return pooled


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


def split_features_labels(
    frame: pd.DataFrame,
    *,
    label_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    mode = label_mode or get_ml_label_mode()
    X = frame[get_feature_columns()].copy()
    if mode == "cross_section" and "cs_label" in frame.columns:
        y = frame["cs_label"].astype(int)
    else:
        y = frame["direction_up"].astype(int)
    return X, y
