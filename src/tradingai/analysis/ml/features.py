from __future__ import annotations

import pandas as pd

from tradingai.analysis.indicators import (
    BB_LOWER,
    BB_MIDDLE,
    BB_UPPER,
    MACD_HISTOGRAM,
    MACD_LINE,
    MACD_SIGNAL,
    MOMENTUM_COLUMN,
    SUPPORTED_INDICATORS,
    compute_indicators,
)
from tradingai.analysis.ml.config import get_ml_include_regime
from tradingai.analysis.ml.regime_features import REGIME_FEATURE_COLUMNS

TECHNICAL_FEATURE_COLUMNS = [
    "rsi",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "atr",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_lower",
    "bb_middle",
    "bb_upper",
    "close_to_sma_20",
    "close_to_sma_50",
    "momentum_roc",
    "volatility",
    "volume_ratio",
    "volume_zscore",
    "rs_spread",
    "rs_line",
]

ML_INDICATORS = sorted(SUPPORTED_INDICATORS)


def get_feature_columns() -> list[str]:
    if get_ml_include_regime():
        return TECHNICAL_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS
    return TECHNICAL_FEATURE_COLUMNS.copy()


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return None
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _row_features(row: pd.Series) -> dict[str, float | None]:
    close = row.get("close")
    sma_20 = row.get("sma_20")
    sma_50 = row.get("sma_50")
    return {
        "rsi": row.get("rsi"),
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": row.get("sma_200"),
        "ema_12": row.get("ema_12"),
        "ema_26": row.get("ema_26"),
        "atr": row.get("atr"),
        "macd": row.get(MACD_LINE),
        "macd_signal": row.get(MACD_SIGNAL),
        "macd_histogram": row.get(MACD_HISTOGRAM),
        "bb_lower": row.get(BB_LOWER),
        "bb_middle": row.get(BB_MIDDLE),
        "bb_upper": row.get(BB_UPPER),
        "close_to_sma_20": _safe_ratio(close, sma_20),
        "close_to_sma_50": _safe_ratio(close, sma_50),
        "momentum_roc": row.get(MOMENTUM_COLUMN),
        "volatility": row.get("volatility"),
        "volume_ratio": row.get("volume_ratio"),
        "volume_zscore": row.get("volume_zscore"),
        "rs_spread": row.get("rs_spread"),
        "rs_line": row.get("rs_line"),
    }


def build_indicator_frame(
    df: pd.DataFrame,
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
) -> pd.DataFrame:
    enriched = compute_indicators(
        df,
        ML_INDICATORS,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark_symbol,
    )
    feature_rows = [_row_features(enriched.iloc[index]) for index in range(len(enriched))]
    features = pd.DataFrame(feature_rows, index=enriched.index)
    result = pd.DataFrame(
        {
            "date": enriched.index,
            "close": enriched["close"].values,
        },
        index=enriched.index,
    )
    for column in TECHNICAL_FEATURE_COLUMNS:
        result[column] = features[column].values
    return result


def latest_feature_vector(
    df: pd.DataFrame,
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
    regime_row: dict[str, float | None] | None = None,
) -> tuple[dict[str, float | None], pd.DataFrame]:
    frame = build_indicator_frame(
        df,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark_symbol,
    )
    latest = frame.iloc[-1]
    snapshot = {
        column: None if pd.isna(latest[column]) else float(latest[column])
        for column in TECHNICAL_FEATURE_COLUMNS
    }
    if get_ml_include_regime():
        if regime_row is None:
            regime_row = _live_regime_row()
        for column in REGIME_FEATURE_COLUMNS:
            value = regime_row.get(column)
            snapshot[column] = None if value is None or pd.isna(value) else float(value)
    return snapshot, frame


def _live_regime_row() -> dict[str, float | None]:
    from tradingai.regime.analysis.historical import macro_features_to_ml_row
    from tradingai.regime.providers.registry import fetch_macro_features

    return macro_features_to_ml_row(fetch_macro_features())


def frame_to_matrix(frame: pd.DataFrame) -> tuple[list[int], list[list[float]]]:
    columns = get_feature_columns()
    indices: list[int] = []
    matrix: list[list[float]] = []
    for index, row in frame.iterrows():
        values = [row[column] for column in columns]
        if any(pd.isna(value) for value in values):
            continue
        indices.append(index)
        matrix.append([float(value) for value in values])
    return indices, matrix


# Backward-compatible alias for callers that expect a module-level list.
FEATURE_COLUMNS = get_feature_columns()
