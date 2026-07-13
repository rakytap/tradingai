from __future__ import annotations

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.data.repositories.stocks import upsert_stock
from tradingai.data.stock_metadata import resolve_stock_metadata


def sync_stocks(tickers: list[str] | None = None) -> dict:
    initialize_schema()
    tickers = tickers or list(get_sync_tickers())
    total = len(tickers)
    log_start("tradingai-sync-stocks", total)
    synced = 0
    errors: list[str] = []

    for index, ticker in enumerate(tickers, start=1):
        try:
            metadata = resolve_stock_metadata(ticker)
            upsert_stock(
                ticker,
                sector=metadata.sector,
                industry=metadata.industry,
                market_cap=metadata.market_cap,
            )
            synced += 1
            detail = metadata.sector or "unknown sector"
            if metadata.industry and metadata.industry != metadata.sector:
                detail = f"{detail} / {metadata.industry}"
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
