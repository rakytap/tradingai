from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_HOST = "postgresql://localhost:5432"
DEFAULT_DATABASE_NAME_PROD = "tradingai"
DEFAULT_DATABASE_NAME_DEV = "tradingai_dev"
DEFAULT_DATABASE_URL = f"{DEFAULT_DATABASE_HOST}/{DEFAULT_DATABASE_NAME_PROD}"
DEFAULT_SYNC_TICKERS = ("AAPL", "MSFT", "NVDA", "AMD", "INTC", "GOOGL", "META", "TSLA", "SPY")

Environment = Literal["dev", "prod"]


@lru_cache
def get_environment() -> Environment:
    raw = (os.getenv("TRADINGAI_ENV") or "prod").strip().lower()
    if raw not in {"dev", "prod"}:
        raise ValueError("TRADINGAI_ENV must be 'dev' or 'prod'.")
    return raw  # type: ignore[return-value]


@lru_cache
def get_database_url() -> str:
    override = os.getenv("TRADINGAI_DATABASE_URL")
    if override:
        return override
    db_name = (
        DEFAULT_DATABASE_NAME_DEV
        if get_environment() == "dev"
        else DEFAULT_DATABASE_NAME_PROD
    )
    return f"{DEFAULT_DATABASE_HOST}/{db_name}"


@lru_cache
def get_sync_tickers() -> tuple[str, ...]:
    raw = os.getenv("TRADINGAI_SYNC_TICKERS")
    if not raw:
        return DEFAULT_SYNC_TICKERS
    return tuple(ticker.strip().upper() for ticker in raw.split(",") if ticker.strip())


@lru_cache
def get_etf_tickers() -> frozenset[str]:
    raw = os.getenv("TRADINGAI_ETF_TICKERS", "")
    if not raw:
        return frozenset()
    return frozenset(ticker.strip().upper() for ticker in raw.split(",") if ticker.strip())
