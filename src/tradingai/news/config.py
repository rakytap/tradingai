import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_NEWS_PROVIDERS = ("yfinance", "sec_edgar", "rss")
DEFAULT_SOCIAL_PROVIDERS = ("reddit",)
DEFAULT_SOCIAL_WEIGHT = 0.45


@lru_cache
def get_news_providers() -> list[str]:
    raw = os.getenv("TRADINGAI_NEWS_PROVIDERS", ",".join(DEFAULT_NEWS_PROVIDERS))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


@lru_cache
def get_social_providers() -> list[str]:
    raw = os.getenv("TRADINGAI_SOCIAL_PROVIDERS", ",".join(DEFAULT_SOCIAL_PROVIDERS))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


@lru_cache
def get_social_weight() -> float:
    raw = os.getenv("TRADINGAI_SOCIAL_WEIGHT", str(DEFAULT_SOCIAL_WEIGHT))
    try:
        weight = float(raw)
    except ValueError:
        return DEFAULT_SOCIAL_WEIGHT
    return max(0.0, min(1.0, weight))


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
