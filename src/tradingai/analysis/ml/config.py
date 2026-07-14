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
MIN_HISTORY_ROWS = 252
MIN_MACRO_HISTORY_ROWS = 252
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


def yfinance_macro_symbol(ticker: str) -> str:
    normalized = ticker.strip().upper()
    return YFINANCE_MACRO_SYMBOLS.get(normalized, normalized)
