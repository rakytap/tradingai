from __future__ import annotations

import yfinance as yf

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.data.repositories.stocks import upsert_stock


def sync_stocks(tickers: list[str] | None = None) -> dict:
    initialize_schema()
    tickers = tickers or list(get_sync_tickers())
    total = len(tickers)
    log_start("tradingai-sync-stocks", total)
    synced = 0
    errors: list[str] = []

    for index, ticker in enumerate(tickers, start=1):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
            industry = info.get("industry")
            upsert_stock(
                ticker,
                sector=sector,
                industry=industry,
                market_cap=info.get("marketCap"),
            )
            synced += 1
            detail = sector or "unknown sector"
            if industry and industry != sector:
                detail = f"{detail} / {industry}"
            log_progress(index, total, ticker, detail)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            log_error(index, total, ticker, str(exc))

    log_done("tradingai-sync-stocks", synced, total, failed=len(errors))

    return {"synced": synced, "errors": errors}


def main() -> None:
    import json

    result = sync_stocks()
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
