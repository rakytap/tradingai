from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_HOST = "postgresql://localhost:5432"
DEFAULT_DATABASE_NAME_PROD = "tradingai"
DEFAULT_DATABASE_NAME_DEV = "tradingai_dev"
DEFAULT_DATABASE_URL = f"{DEFAULT_DATABASE_HOST}/{DEFAULT_DATABASE_NAME_PROD}"
# Small fallback if the packaged universe file is missing.
DEFAULT_SYNC_TICKERS = ("AAPL", "MSFT", "NVDA", "AMD", "INTC", "GOOGL", "META", "TSLA")
DEFAULT_UNIVERSE_FILE = Path(__file__).resolve().parent / "universe" / "us_equities_1000.txt"
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")

Environment = Literal["dev", "prod"]


def _parse_ticker_line(raw: str) -> str | None:
    line = raw.split("#", 1)[0].strip().upper().replace(" ", "")
    if not line:
        return None
    if not _TICKER_RE.fullmatch(line):
        return None
    return line


def load_tickers_from_file(path: Path | str) -> tuple[str, ...]:
    """Load unique equity tickers from a one-symbol-per-line text file."""
    text = Path(path).read_text(encoding="utf-8")
    seen: set[str] = set()
    out: list[str] = []
    for raw in text.splitlines():
        ticker = _parse_ticker_line(raw)
        if ticker is None or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return tuple(out)


def get_default_universe_path() -> Path:
    raw = os.getenv("TRADINGAI_SYNC_TICKERS_FILE")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    return DEFAULT_UNIVERSE_FILE


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
    """Equity universe for tradingai-sync-stocks (and downstream DB membership).

    Precedence:
    1. TRADINGAI_SYNC_TICKERS (comma-separated override)
    2. TRADINGAI_SYNC_TICKERS_FILE, or packaged us_equities_1000.txt
    3. DEFAULT_SYNC_TICKERS fallback
    """
    raw = os.getenv("TRADINGAI_SYNC_TICKERS")
    if raw and raw.strip():
        return tuple(
            ticker.strip().upper()
            for ticker in raw.split(",")
            if ticker.strip()
        )
    path = get_default_universe_path()
    if path.is_file():
        tickers = load_tickers_from_file(path)
        if tickers:
            return tickers
    return DEFAULT_SYNC_TICKERS


@lru_cache
def get_etf_tickers() -> frozenset[str]:
    raw = os.getenv("TRADINGAI_ETF_TICKERS", "")
    if not raw:
        return frozenset()
    return frozenset(ticker.strip().upper() for ticker in raw.split(",") if ticker.strip())
