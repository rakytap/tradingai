from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_WEIGHTS = (0.45, 0.35, 0.20)
DEFAULT_HORIZONS = (1, 5, 20)

HORIZON_SCALES = {
    1: 1.0,
    5: 1.2,
    20: 0.9,
}

HORIZON_RETURN_MULTIPLIERS = {
    1: 0.25,
    5: 1.90,
    20: 3.85,
}


@lru_cache
def get_fusion_weights() -> tuple[float, float, float]:
    raw = os.getenv("TRADINGAI_PREDICT_WEIGHTS", ",".join(str(w) for w in DEFAULT_WEIGHTS))
    parts = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(parts) != 3:
        return DEFAULT_WEIGHTS
    total = sum(parts)
    if total <= 0:
        return DEFAULT_WEIGHTS
    return (parts[0] / total, parts[1] / total, parts[2] / total)


@lru_cache
def get_prediction_horizons() -> tuple[int, ...]:
    raw = os.getenv("TRADINGAI_PREDICT_HORIZONS", ",".join(str(h) for h in DEFAULT_HORIZONS))
    horizons = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    return horizons or DEFAULT_HORIZONS
