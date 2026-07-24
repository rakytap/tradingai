import numpy as np
import pandas as pd
import pytest

from tradingai.analysis.ml.config import get_ml_include_regime
from tradingai.analysis.ml.features import (
    TECHNICAL_FEATURE_COLUMNS,
    build_indicator_frame,
    get_feature_columns,
    latest_feature_vector,
)


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
    for column in TECHNICAL_FEATURE_COLUMNS:
        assert column in frame.columns
    assert frame["rsi"].notna().any()


def test_latest_feature_vector_returns_complete_snapshot(monkeypatch: pytest.MonkeyPatch):
    from tradingai.analysis.ml import features as features_module

    monkeypatch.setenv("TRADINGAI_ML_INCLUDE_REGIME", "true")
    get_ml_include_regime.cache_clear()
    monkeypatch.setattr(
        features_module,
        "_live_regime_row",
        lambda: {
            "regime_vix": 16.0,
            "regime_vix_pct_1y": 40.0,
            "regime_spy_return_6m_pct": 8.0,
            "regime_spy_close_to_sma200": 1.05,
            "regime_hyg_lqd_ratio": 0.68,
            "regime_score": 0.8,
            "regime_high_vol": 0.0,
            "regime_bull_trend": 1.0,
        },
    )

    snapshot, frame = latest_feature_vector(
        _sample_ohlcv(),
        benchmark_df=_sample_benchmark(),
        benchmark_symbol="SPY",
        sector_df=_sample_benchmark(),
    )
    assert set(snapshot.keys()) == set(get_feature_columns())
    assert len(frame) == 300
    assert snapshot["rsi"] is not None
    assert snapshot["regime_score"] == 0.8
    assert snapshot["sector_rs_spread"] is not None
