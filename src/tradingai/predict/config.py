from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_WEIGHTS = (0.45, 0.35, 0.20)
DEFAULT_HORIZONS = (3, 5, 8)
DEFAULT_FUSION_TUNING_PATH = Path("./data/fusion_weights.json")

HORIZON_SCALES = {
    3: 1.1,
    5: 1.2,
    8: 1.0,
}

HORIZON_RETURN_MULTIPLIERS = {
    3: 1.0,
    5: 1.90,
    8: 2.5,
}


@lru_cache
def get_predict_backend() -> str:
    return os.getenv("TRADINGAI_PREDICT_BACKEND", "fusion").lower().strip()


@lru_cache
def get_ml_model_name() -> str:
    return os.getenv("TRADINGAI_ML_MODEL", "logistic").lower().strip()


@lru_cache
def get_ml_horizon_days() -> int:
    raw = os.getenv("TRADINGAI_ML_HORIZON", "5")
    try:
        return int(raw.strip())
    except ValueError:
        return 5


@lru_cache
def is_fusion_tuning_enabled() -> bool:
    raw = os.getenv("TRADINGAI_FUSION_TUNING_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@lru_cache
def get_fusion_tuning_path() -> Path:
    raw = os.getenv("TRADINGAI_FUSION_TUNING_PATH", str(DEFAULT_FUSION_TUNING_PATH))
    return Path(raw)


@lru_cache
def load_fusion_tuning() -> dict | None:
    if not is_fusion_tuning_enabled():
        return None

    path = get_fusion_tuning_path()
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if "weights" not in payload:
        return None
    return payload


def _weights_from_env() -> tuple[float, float, float]:
    raw = os.getenv("TRADINGAI_PREDICT_WEIGHTS", ",".join(str(w) for w in DEFAULT_WEIGHTS))
    parts = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(parts) != 3:
        return DEFAULT_WEIGHTS
    total = sum(parts)
    if total <= 0:
        return DEFAULT_WEIGHTS
    return (parts[0] / total, parts[1] / total, parts[2] / total)


def _weights_from_tuning(payload: dict) -> tuple[float, float, float] | None:
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        return None
    try:
        technical = float(weights["technical"])
        news = float(weights["news"])
        regime = float(weights["regime"])
    except (KeyError, TypeError, ValueError):
        return None
    total = technical + news + regime
    if total <= 0:
        return None
    return (technical / total, news / total, regime / total)


@lru_cache
def get_fusion_weights() -> tuple[float, float, float]:
    tuning = load_fusion_tuning()
    if tuning is not None:
        tuned = _weights_from_tuning(tuning)
        if tuned is not None:
            return tuned
    return _weights_from_env()


@lru_cache
def get_horizon_scales() -> dict[int, float]:
    scales = dict(HORIZON_SCALES)
    tuning = load_fusion_tuning()
    if tuning is None:
        return scales

    tuned_scales = tuning.get("horizon_scales")
    if not isinstance(tuned_scales, dict):
        return scales

    for key, value in tuned_scales.items():
        try:
            scales[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return scales


def clear_fusion_config_cache() -> None:
    get_predict_backend.cache_clear()
    get_ml_model_name.cache_clear()
    get_ml_horizon_days.cache_clear()
    is_fusion_tuning_enabled.cache_clear()
    get_fusion_tuning_path.cache_clear()
    load_fusion_tuning.cache_clear()
    get_fusion_weights.cache_clear()
    get_horizon_scales.cache_clear()


@lru_cache
def get_prediction_horizons() -> tuple[int, ...]:
    raw = os.getenv("TRADINGAI_PREDICT_HORIZONS", ",".join(str(h) for h in DEFAULT_HORIZONS))
    horizons = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    return horizons or DEFAULT_HORIZONS
