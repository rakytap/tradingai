from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingai.analysis.ml.config import get_macro_sync_tickers, get_ml_include_regime
from tradingai.analysis.ml.features import (
    TECHNICAL_FEATURE_COLUMNS,
    get_feature_columns,
    latest_feature_vector,
)
from tradingai.analysis.ml.regime_features import REGIME_FEATURE_COLUMNS
from tradingai.regime.analysis.historical import (
    build_regime_feature_frame,
    macro_features_as_of,
    macro_features_to_ml_row,
)
from tradingai.regime.analysis.classifier import classify_regime


def _macro_series(start: str, values: list[float]) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index)


def _bull_spy_series(rows: int = 300) -> pd.Series:
    return _macro_series("2023-01-01", np.linspace(400, 520, rows).tolist())


def _calm_vix_series(rows: int = 300) -> pd.Series:
    values = np.linspace(12.0, 22.0, rows).tolist()
    values[-1] = 15.0
    return _macro_series("2023-01-01", values)


def _hyg_lqd_series(rows: int = 300) -> tuple[pd.Series, pd.Series]:
    hyg = _macro_series("2023-01-01", np.linspace(75, 78, rows).tolist())
    lqd = _macro_series("2023-01-01", np.linspace(110, 112, rows).tolist())
    return hyg, lqd


def test_macro_features_as_of_bull_regime():
    spy = _bull_spy_series()
    vix = _calm_vix_series()
    hyg, lqd = _hyg_lqd_series()
    as_of = spy.index[-1].date()

    features = macro_features_as_of(
        spy=spy,
        vix=vix,
        hyg=hyg,
        lqd=lqd,
        as_of=as_of,
        sources_used=["test"],
    )
    assert classify_regime(features) == "bull"
    row = macro_features_to_ml_row(features)
    assert row["regime_bull_trend"] == 1.0
    assert row["regime_high_vol"] == 0.0
    assert row["regime_score"] == pytest.approx(0.8)


def test_macro_features_as_of_handles_timezone_aware_index():
    spy = _bull_spy_series()
    vix = _calm_vix_series()
    vix.index = vix.index.tz_localize("America/Chicago")
    hyg, lqd = _hyg_lqd_series()
    as_of = spy.index[-1].date()

    features = macro_features_as_of(
        spy=spy,
        vix=vix,
        hyg=hyg,
        lqd=lqd,
        as_of=as_of,
        sources_used=["test"],
    )
    assert features.vix == pytest.approx(15.0)


def test_build_regime_feature_frame_has_expected_columns():
    spy = _bull_spy_series()
    vix = _calm_vix_series()
    hyg, lqd = _hyg_lqd_series()
    frame = build_regime_feature_frame(
        spy.index[-60:],
        {"SPY": spy, "VIX": vix, "HYG": hyg, "LQD": lqd},
    )
    for column in REGIME_FEATURE_COLUMNS:
        assert column in frame.columns
    assert len(frame) == 60
    assert frame["regime_vix"].notna().all()


def test_get_feature_columns_includes_regime_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRADINGAI_ML_INCLUDE_REGIME", raising=False)
    get_ml_include_regime.cache_clear()
    columns = get_feature_columns()
    assert TECHNICAL_FEATURE_COLUMNS == columns[: len(TECHNICAL_FEATURE_COLUMNS)]
    assert REGIME_FEATURE_COLUMNS == columns[len(TECHNICAL_FEATURE_COLUMNS) :]


def test_get_feature_columns_excludes_regime_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRADINGAI_ML_INCLUDE_REGIME", "false")
    get_ml_include_regime.cache_clear()
    assert get_feature_columns() == TECHNICAL_FEATURE_COLUMNS


def test_load_macro_histories_falls_back_to_yfinance(monkeypatch: pytest.MonkeyPatch):
    from tradingai.analysis.ml import macro_history

    def _empty_db(_ticker: str, *, database_url: str | None = None) -> pd.Series:
        return pd.Series(dtype=float)

    def _fetch_yfinance(ticker: str, *, period: str = "2y") -> pd.Series:
        values = {
            "SPY": _bull_spy_series(),
            "VIX": _calm_vix_series(),
            "HYG": _hyg_lqd_series()[0],
            "LQD": _hyg_lqd_series()[1],
            "TNX": _macro_series("2023-01-01", np.linspace(3.5, 4.5, 300).tolist()),
        }
        return values.get(ticker, pd.Series(dtype=float))

    monkeypatch.setattr(macro_history, "_close_series_from_db", _empty_db)
    monkeypatch.setattr(macro_history, "_fetch_yfinance_close", _fetch_yfinance)

    histories = macro_history.load_macro_histories()
    assert set(histories.keys()) >= {"SPY", "VIX", "HYG", "LQD"}
    assert len(histories["SPY"]) >= 252


def test_join_regime_features_in_build_ticker_dataset(monkeypatch: pytest.MonkeyPatch):
    from tradingai.analysis.ml import dataset

    monkeypatch.setenv("TRADINGAI_ML_INCLUDE_REGIME", "true")
    get_ml_include_regime.cache_clear()

    ohlcv_index = pd.date_range("2023-01-01", periods=300, freq="D")
    ohlcv = pd.DataFrame(
        {
            "open": np.linspace(100, 150, 300),
            "high": np.linspace(101, 151, 300),
            "low": np.linspace(99, 149, 300),
            "close": np.linspace(100, 150, 300),
            "volume": np.linspace(900_000, 1_100_000, 300).astype(int),
        },
        index=ohlcv_index,
    )

    spy = _bull_spy_series(300)
    vix = _calm_vix_series(300)
    hyg, lqd = _hyg_lqd_series(300)
    regime_frame = build_regime_feature_frame(ohlcv_index, {"SPY": spy, "VIX": vix, "HYG": hyg, "LQD": lqd})

    monkeypatch.setattr(dataset, "load_ticker_ohlcv", lambda ticker, **kwargs: ohlcv)

    frame = dataset.build_ticker_dataset(
        "AAPL",
        horizon_days=5,
        benchmark_df=ohlcv,
        regime_frame=regime_frame,
    )
    assert not frame.empty
    for column in REGIME_FEATURE_COLUMNS:
        assert column in frame.columns


def test_latest_feature_vector_includes_regime_columns(monkeypatch: pytest.MonkeyPatch):
    from tradingai.analysis.ml import features as features_module

    monkeypatch.setenv("TRADINGAI_ML_INCLUDE_REGIME", "true")
    get_ml_include_regime.cache_clear()

    regime_row = {
        "regime_vix": 16.0,
        "regime_vix_pct_1y": 40.0,
        "regime_spy_return_6m_pct": 8.0,
        "regime_spy_close_to_sma200": 1.05,
        "regime_hyg_lqd_ratio": 0.68,
        "regime_score": 0.8,
        "regime_high_vol": 0.0,
        "regime_bull_trend": 1.0,
    }
    monkeypatch.setattr(features_module, "_live_regime_row", lambda: regime_row)

    index = pd.date_range("2023-01-01", periods=300, freq="D")
    close = pd.Series(np.linspace(100, 180, 300), index=index)
    ohlcv = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.full(300, 1_000_000),
        },
        index=index,
    )
    benchmark = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 1,
            "close": close * 4,
            "volume": np.full(300, 5_000_000),
        },
        index=index,
    )

    snapshot, _ = latest_feature_vector(ohlcv, benchmark_df=benchmark, benchmark_symbol="SPY")
    assert set(snapshot.keys()) == set(get_feature_columns())
    assert snapshot["regime_score"] == 0.8


def test_get_macro_sync_tickers_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRADINGAI_MACRO_SYNC_TICKERS", raising=False)
    get_macro_sync_tickers.cache_clear()
    assert get_macro_sync_tickers() == ("SPY", "VIX", "HYG", "LQD")
