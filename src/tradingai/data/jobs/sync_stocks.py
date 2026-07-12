from __future__ import annotations

import yfinance as yf

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.stocks import upsert_stock


def sync_stocks(tickers: list[str] | None = None) -> dict:
    initialize_schema()
    tickers = tickers or list(get_sync_tickers())
    synced = 0
    errors: list[str] = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            upsert_stock(
                ticker,
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=info.get("marketCap"),
            )
            synced += 1
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    return {"synced": synced, "errors": errors}


def main() -> None:
    import json

    print(json.dumps(sync_stocks(), indent=2))


if __name__ == "__main__":
    main()
