from __future__ import annotations

import numpy as np

from tradingai.analysis.ml.artifacts import (
    artifact_exists,
    get_manifest_version,
    load_artifact,
    validate_manifest_features,
)
from tradingai.analysis.ml.config import get_default_ml_model, get_ml_benchmark, get_model_dir
from tradingai.analysis.ml.features import get_feature_columns, latest_feature_vector
from tradingai.analysis.ml.ohlcv import load_ticker_ohlcv
from tradingai.providers.registry import get_provider


def _fetch_live_ohlcv(ticker: str, *, period: str = "2y"):
    provider = get_provider()
    symbol_df = provider.get_ohlcv(ticker.upper(), period, "1d")
    benchmark = get_ml_benchmark()
    benchmark_df = provider.get_ohlcv(benchmark, period, "1d")
    return symbol_df, benchmark_df, benchmark


def _fetch_db_ohlcv(ticker: str):
    symbol_df = load_ticker_ohlcv(ticker)
    benchmark = get_ml_benchmark()
    benchmark_df = load_ticker_ohlcv(benchmark)
    if symbol_df.empty:
        raise ValueError(f"No price history in database for {ticker.upper()}.")
    if benchmark_df.empty:
        raise ValueError(f"No benchmark history in database for {benchmark}.")
    return symbol_df, benchmark_df, benchmark


def predict_direction(
    ticker: str,
    *,
    model_name: str | None = None,
    horizon_days: int = 5,
    model_dir=None,
    use_database: bool = False,
) -> dict:
    selected_model = (model_name or get_default_ml_model()).lower().strip()
    base_dir = model_dir or get_model_dir()
    if not artifact_exists(selected_model, horizon_days=horizon_days, model_dir=base_dir):
        raise FileNotFoundError(
            f"Trained artifact missing for model '{selected_model}' at horizon {horizon_days}d."
        )

    if use_database:
        symbol_df, benchmark_df, benchmark = _fetch_db_ohlcv(ticker)
    else:
        symbol_df, benchmark_df, benchmark = _fetch_live_ohlcv(ticker)

    snapshot, _ = latest_feature_vector(
        symbol_df,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark,
    )
    feature_columns = get_feature_columns()
    validate_manifest_features(feature_columns, model_dir=base_dir)
    if any(snapshot.get(column) is None for column in feature_columns):
        raise ValueError("Latest feature vector contains missing values; need more price history.")

    X = np.array([[snapshot[column] for column in feature_columns]], dtype=float)
    predictor = load_artifact(selected_model, horizon_days=horizon_days, model_dir=base_dir)
    probabilities = predictor.predict_proba(X)[0]
    classes = getattr(predictor, "classes_", None)
    if classes is not None and len(classes) == 2:
        up_index = int(np.where(classes == 1)[0][0])
        probability_up = float(probabilities[up_index])
    else:
        probability_up = float(probabilities[-1])

    directional_score = max(-1.0, min(1.0, probability_up * 2.0 - 1.0))
    return {
        "ticker": ticker.upper(),
        "model": selected_model,
        "horizon_days": horizon_days,
        "probability_up": round(probability_up, 4),
        "directional_score": round(directional_score, 4),
        "model_version": get_manifest_version(model_dir=base_dir),
        "feature_snapshot": snapshot,
    }
