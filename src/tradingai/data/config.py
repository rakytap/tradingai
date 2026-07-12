from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql://localhost:5432/tradingai"
DEFAULT_SYNC_TICKERS = ("AAPL", "MSFT", "NVDA", "AMD", "INTC", "GOOGL", "META", "TSLA", "SPY")


@lru_cache
def get_database_url() -> str:
    return os.getenv("TRADINGAI_DATABASE_URL", DEFAULT_DATABASE_URL)


@lru_cache
def get_sync_tickers() -> tuple[str, ...]:
    raw = os.getenv("TRADINGAI_SYNC_TICKERS")
    if not raw:
        return DEFAULT_SYNC_TICKERS
    return tuple(ticker.strip().upper() for ticker in raw.split(",") if ticker.strip())
