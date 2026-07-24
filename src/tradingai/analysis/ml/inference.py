from __future__ import annotations

import numpy as np

from tradingai.analysis.ml.artifacts import (
    artifact_exists,
    get_manifest_version,
    load_artifact,
    load_manifest,
    validate_manifest_features,
)
from tradingai.analysis.ml.config import get_default_ml_model, get_ml_benchmark, get_model_dir
from tradingai.analysis.ml.features import get_feature_columns, latest_feature_vector
from tradingai.analysis.ml.ohlcv import load_ticker_ohlcv
from tradingai.analysis.ml.sector_models import resolve_model_dir_for_sector
from tradingai.data.repositories.stocks import get_stock
from tradingai.providers.registry import get_provider
from tradingai.regime.sector_proxies import normalize_sector_label, sector_regime_proxy


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


def _resolve_sector(ticker: str) -> str | None:
    try:
        stock = get_stock(ticker)
    except Exception:
        return None
    if not stock:
        return None
    return normalize_sector_label(stock.get("sector")) or stock.get("sector")


def predict_direction(
    ticker: str,
    *,
    model_name: str | None = None,
    horizon_days: int = 5,
    model_dir=None,
    use_database: bool = False,
    sector: str | None = None,
) -> dict:
    selected_model = (model_name or get_default_ml_model()).lower().strip()
    base_dir = model_dir or get_model_dir()
    resolved_sector = sector if sector is not None else _resolve_sector(ticker)

    try:
        artifact_root, model_source = resolve_model_dir_for_sector(
            resolved_sector,
            model_name=selected_model,
            horizon_days=horizon_days,
            model_dir=base_dir,
        )
    except FileNotFoundError:
        # Explicit global-only path when caller passed model_dir as artifact root.
        if not artifact_exists(selected_model, horizon_days=horizon_days, model_dir=base_dir):
            raise
        artifact_root, model_source = base_dir, "global"

    if use_database:
        symbol_df, benchmark_df, benchmark = _fetch_db_ohlcv(ticker)
    else:
        symbol_df, benchmark_df, benchmark = _fetch_live_ohlcv(ticker)

    proxy = sector_regime_proxy(resolved_sector)
    try:
        if use_database:
            sector_df = load_ticker_ohlcv(proxy)
        else:
            sector_df = get_provider().get_ohlcv(proxy, "2y", "1d")
    except Exception:
        sector_df = benchmark_df

    snapshot, _ = latest_feature_vector(
        symbol_df,
        benchmark_df=benchmark_df,
        benchmark_symbol=benchmark,
        sector_df=sector_df if sector_df is not None and not sector_df.empty else benchmark_df,
    )
    feature_columns = get_feature_columns()
    validate_manifest_features(feature_columns, model_dir=artifact_root)
    if any(snapshot.get(column) is None for column in feature_columns):
        raise ValueError("Latest feature vector contains missing values; need more price history.")

    X = np.array([[snapshot[column] for column in feature_columns]], dtype=float)
    predictor = load_artifact(selected_model, horizon_days=horizon_days, model_dir=artifact_root)
    probabilities = predictor.predict_proba(X)[0]
    classes = getattr(predictor, "classes_", None)
    if classes is not None and len(classes) == 2:
        up_index = int(np.where(classes == 1)[0][0])
        probability_up = float(probabilities[up_index])
    else:
        probability_up = float(probabilities[-1])

    directional_score = max(-1.0, min(1.0, probability_up * 2.0 - 1.0))
    manifest = load_manifest(model_dir=artifact_root) or {}
    label_mode = manifest.get("label_mode") or "absolute"
    return {
        "ticker": ticker.upper(),
        "model": selected_model,
        "horizon_days": horizon_days,
        "probability_up": round(probability_up, 4),
        "directional_score": round(directional_score, 4),
        "cs_prob": round(probability_up, 4) if label_mode == "cross_section" else None,
        "model_version": get_manifest_version(model_dir=artifact_root),
        "feature_snapshot": snapshot,
        "label_mode": label_mode,
        "model_source": model_source,
        "sector": resolved_sector,
        "rank_context": {
            "sector": resolved_sector,
            "sector_proxy": proxy,
            "model_dir": str(artifact_root),
        },
    }
