from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_NEWS_PROVIDERS = ("yfinance", "sec_edgar", "rss")
DEFAULT_SOCIAL_PROVIDERS = ("reddit",)
DEFAULT_SOCIAL_WEIGHT = 0.45
DEFAULT_NEWS_SCORING_PATH = Path("./data/news_scoring.json")

DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "news": 1.0,
    "press": 1.1,
    "analyst": 1.4,
    "filings": 0.9,
    "earnings": 1.3,
    "transcript": 1.0,
    "social": DEFAULT_SOCIAL_WEIGHT,
}

DEFAULT_RECENCY: dict[str, float] = {
    "half_life_days": 7.0,
    "floor": 0.4,
    "missing_date_weight": 0.6,
}


def default_news_scoring_params() -> dict[str, Any]:
    return {
        "source_weights": dict(DEFAULT_SOURCE_WEIGHTS),
        "recency": dict(DEFAULT_RECENCY),
    }


@lru_cache
def get_news_providers() -> list[str]:
    raw = os.getenv("TRADINGAI_NEWS_PROVIDERS", ",".join(DEFAULT_NEWS_PROVIDERS))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


@lru_cache
def get_social_providers() -> list[str]:
    raw = os.getenv("TRADINGAI_SOCIAL_PROVIDERS", ",".join(DEFAULT_SOCIAL_PROVIDERS))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _env_social_weight() -> float:
    raw = os.getenv("TRADINGAI_SOCIAL_WEIGHT", str(DEFAULT_SOCIAL_WEIGHT))
    try:
        weight = float(raw)
    except ValueError:
        return DEFAULT_SOCIAL_WEIGHT
    return max(0.0, min(1.0, weight))


@lru_cache
def get_social_weight() -> float:
    params = get_news_scoring_params()
    social = params["source_weights"].get("social")
    if social is not None:
        return max(0.0, min(1.0, float(social)))
    return _env_social_weight()


@lru_cache
def is_news_scoring_enabled() -> bool:
    raw = os.getenv("TRADINGAI_NEWS_SCORING_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@lru_cache
def get_news_scoring_path() -> Path:
    raw = os.getenv("TRADINGAI_NEWS_SCORING_PATH", str(DEFAULT_NEWS_SCORING_PATH))
    return Path(raw)


@lru_cache
def load_news_scoring() -> dict | None:
    if not is_news_scoring_enabled():
        return None
    path = get_news_scoring_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if "source_weights" not in payload and "recency" not in payload:
        return None
    return payload


def _merge_scoring_params(payload: dict | None) -> dict[str, Any]:
    params = default_news_scoring_params()
    params["source_weights"]["social"] = _env_social_weight()
    if not payload:
        return params

    source_weights = payload.get("source_weights")
    if isinstance(source_weights, dict):
        for key in DEFAULT_SOURCE_WEIGHTS:
            if key in source_weights:
                try:
                    params["source_weights"][key] = float(source_weights[key])
                except (TypeError, ValueError):
                    continue

    recency = payload.get("recency")
    if isinstance(recency, dict):
        for key in DEFAULT_RECENCY:
            if key in recency:
                try:
                    params["recency"][key] = float(recency[key])
                except (TypeError, ValueError):
                    continue
    return params


@lru_cache
def get_news_scoring_params() -> dict[str, Any]:
    return _merge_scoring_params(load_news_scoring())


def save_news_scoring_params(result: dict, path: Path | None = None) -> Path:
    target = path or get_news_scoring_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tuned_at": result.get("tuned_at"),
        "metric": result.get("metric", "brier"),
        "source_weights": result["source_weights"],
        "recency": result["recency"],
        "sample_sizes": result.get("sample_sizes"),
        "baseline_validation": result.get("baseline_validation"),
        "tuned_validation": result.get("tuned_validation"),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    clear_news_scoring_cache()
    return target


def clear_news_scoring_cache() -> None:
    get_news_providers.cache_clear()
    get_social_providers.cache_clear()
    get_social_weight.cache_clear()
    is_news_scoring_enabled.cache_clear()
    get_news_scoring_path.cache_clear()
    load_news_scoring.cache_clear()
    get_news_scoring_params.cache_clear()


@lru_cache
def get_reddit_user_agent() -> str:
    return os.getenv(
        "REDDIT_USER_AGENT",
        "tradingai-news-mcp/0.1 by tradingai",
    )


@lru_cache
def get_reddit_client_id() -> str | None:
    return os.getenv("REDDIT_CLIENT_ID")


@lru_cache
def get_reddit_client_secret() -> str | None:
    return os.getenv("REDDIT_CLIENT_SECRET")


@lru_cache
def get_x_bearer_token() -> str | None:
    return os.getenv("X_BEARER_TOKEN")


@lru_cache
def get_sec_identity() -> str | None:
    return os.getenv("SEC_IDENTITY")


@lru_cache
def get_news_api_key() -> str | None:
    return os.getenv("NEWS_API_KEY")


@lru_cache
def get_finnhub_api_key() -> str | None:
    return os.getenv("FINNHUB_API_KEY")
