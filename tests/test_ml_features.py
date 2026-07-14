import numpy as np
import pandas as pd

from tradingai.analysis.ml.features import FEATURE_COLUMNS, build_indicator_frame, latest_feature_vector


def _sample_ohlcv(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = pd.Series(np.linspace(100, 180, rows), index=index)
    volume = np.full(rows, 1_000_000)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _sample_benchmark(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = pd.Series(np.linspace(400, 450, rows), index=index)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(rows, 5_000_000),
        },
        index=index,
    )


def test_build_indicator_frame_has_feature_columns():
    frame = build_indicator_frame(
        _sample_ohlcv(),
        benchmark_df=_sample_benchmark(),
        benchmark_symbol="SPY",
    )
    for column in FEATURE_COLUMNS:
        assert column in frame.columns
    assert frame["rsi"].notna().any()


def test_latest_feature_vector_returns_complete_snapshot():
    snapshot, frame = latest_feature_vector(
        _sample_ohlcv(),
        benchmark_df=_sample_benchmark(),
        benchmark_symbol="SPY",
    )
    assert set(snapshot.keys()) == set(FEATURE_COLUMNS)
    assert len(frame) == 300
    assert snapshot["rsi"] is not None
