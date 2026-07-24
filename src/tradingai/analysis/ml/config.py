from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_DIR = Path("./data/models")
DEFAULT_ML_MODEL = "logistic"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_DIRECTION_THRESHOLD = 0.5
DEFAULT_MACRO_TICKERS = ("SPY", "VIX", "HYG", "LQD")
DEFAULT_LABEL_MODE = "cross_section"
MIN_HISTORY_ROWS = 252
MIN_MACRO_HISTORY_ROWS = 252
MIN_CS_PEERS = 5
MIN_SECTOR_TICKERS = 8
TRAIN_FRACTION = 0.8

YFINANCE_MACRO_SYMBOLS = {
    "VIX": "^VIX",
    "TNX": "^TNX",
}


@lru_cache
def get_model_dir() -> Path:
    raw = os.getenv("TRADINGAI_ML_MODEL_DIR", str(DEFAULT_MODEL_DIR))
    return Path(raw)


@lru_cache
def get_default_ml_model() -> str:
    return os.getenv("TRADINGAI_ML_MODEL", DEFAULT_ML_MODEL).lower().strip()


@lru_cache
def get_ml_benchmark() -> str:
    return os.getenv("TRADINGAI_ML_BENCHMARK", DEFAULT_BENCHMARK).upper().strip()


@lru_cache
def get_direction_threshold() -> float:
    raw = os.getenv("TRADINGAI_DIRECTION_THRESHOLD", str(DEFAULT_DIRECTION_THRESHOLD))
    try:
        value = float(raw.strip())
    except ValueError:
        return DEFAULT_DIRECTION_THRESHOLD
    return min(1.0, max(0.0, value))


@lru_cache
def get_macro_sync_tickers() -> tuple[str, ...]:
    raw = os.getenv("TRADINGAI_MACRO_SYNC_TICKERS")
    if raw:
        tickers = tuple(ticker.strip().upper() for ticker in raw.split(",") if ticker.strip())
        if tickers:
            return tickers
    return DEFAULT_MACRO_TICKERS


@lru_cache
def get_ml_include_regime() -> bool:
    raw = os.getenv("TRADINGAI_ML_INCLUDE_REGIME", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@lru_cache
def get_ml_label_mode() -> str:
    raw = os.getenv("TRADINGAI_ML_LABEL_MODE", DEFAULT_LABEL_MODE).strip().lower()
    if raw in {"cross_section", "cross-section", "cs", "relative"}:
        return "cross_section"
    return "absolute"


@lru_cache
def get_ml_allow_global_fallback() -> bool:
    raw = os.getenv("TRADINGAI_ML_ALLOW_GLOBAL_FALLBACK", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache
def is_l5_ranking_enabled() -> bool:
    raw = os.getenv("TRADINGAI_L5_RANKING", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def yfinance_macro_symbol(ticker: str) -> str:
    normalized = ticker.strip().upper()
    return YFINANCE_MACRO_SYMBOLS.get(normalized, normalized)


def clear_ml_config_cache() -> None:
    get_model_dir.cache_clear()
    get_default_ml_model.cache_clear()
    get_ml_benchmark.cache_clear()
    get_direction_threshold.cache_clear()
    get_macro_sync_tickers.cache_clear()
    get_ml_include_regime.cache_clear()
    get_ml_label_mode.cache_clear()
    get_ml_allow_global_fallback.cache_clear()
    is_l5_ranking_enabled.cache_clear()
