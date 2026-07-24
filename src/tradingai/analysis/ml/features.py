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
    RS_LOOKBACK,
    SUPPORTED_INDICATORS,
    compute_indicators,
)
from tradingai.analysis.ml.config import get_ml_include_regime
from tradingai.analysis.ml.regime_features import REGIME_FEATURE_COLUMNS

SECTOR_EXCESS_LOOKBACK = 21

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
    "sector_rs_spread",
    "sector_rs_line",
    "sector_excess_1m",
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


def attach_sector_relative_features(
    frame: pd.DataFrame,
    sector_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add sector ETF relative-strength columns onto an indicator feature frame."""
    result = frame.copy()
    if sector_df is None or sector_df.empty or "close" not in result.columns:
        result["sector_rs_spread"] = pd.NA
        result["sector_rs_line"] = pd.NA
        result["sector_excess_1m"] = pd.NA
        return result

    if "date" in result.columns:
        work_index = pd.to_datetime(result["date"])
        sector_close = sector_df["close"].reindex(work_index, method="ffill")
        symbol_close = result["close"].reset_index(drop=True)
        sector_close = pd.Series(sector_close.to_numpy(), index=symbol_close.index)
    else:
        sector_close = sector_df["close"].reindex(result.index, method="ffill")
        symbol_close = result["close"]

    symbol_return = symbol_close.pct_change(RS_LOOKBACK)
    sector_return = sector_close.pct_change(RS_LOOKBACK)
    result["sector_rs_spread"] = (symbol_return - sector_return).to_numpy() * 100

    first = float(symbol_close.iloc[0]) if len(symbol_close) else float("nan")
    sector_first = float(sector_close.iloc[0]) if len(sector_close) else float("nan")
    if first and sector_first and first == first and sector_first == sector_first:
        symbol_base = symbol_close / first
        sector_base = sector_close / sector_first
        result["sector_rs_line"] = (symbol_base / sector_base).to_numpy() * 100
    else:
        result["sector_rs_line"] = pd.NA

    symbol_1m = symbol_close.pct_change(SECTOR_EXCESS_LOOKBACK)
    sector_1m = sector_close.pct_change(SECTOR_EXCESS_LOOKBACK)
    result["sector_excess_1m"] = (symbol_1m - sector_1m).to_numpy() * 100
    return result


def build_indicator_frame(
    df: pd.DataFrame,
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
    sector_df: pd.DataFrame | None = None,
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
        if column.startswith("sector_"):
            continue
        result[column] = features[column].values
    result = attach_sector_relative_features(result, sector_df)
    return result


def latest_feature_vector(
    df: pd.DataFrame,
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_symbol: str = "SPY",
    sector_df: pd.DataFrame | None = None,
    regime_row: dict[str, float | None] | None = None,
) -> tuple[dict[str, float | None], pd.DataFrame]:
    frame = build_indicator_frame(
        df,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark_symbol,
        sector_df=sector_df,
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
