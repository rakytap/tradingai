from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_DIR = Path("./data/models")
DEFAULT_ML_MODEL = "logistic"
DEFAULT_BENCHMARK = "SPY"
MIN_HISTORY_ROWS = 252
TRAIN_FRACTION = 0.8


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
