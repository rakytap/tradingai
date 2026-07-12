import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_REGIME_PROVIDERS = ("yfinance", "fred")


@lru_cache
def get_regime_providers() -> list[str]:
    raw = os.getenv("TRADINGAI_REGIME_PROVIDERS", ",".join(DEFAULT_REGIME_PROVIDERS))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


@lru_cache
def get_fred_api_key() -> str | None:
    return os.getenv("FRED_API_KEY")
