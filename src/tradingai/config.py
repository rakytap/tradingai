import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache
def get_data_provider_name() -> str:
    return os.getenv("TRADINGAI_DATA_PROVIDER", "yfinance").lower()
