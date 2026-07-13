from __future__ import annotations

from datetime import date

from tradingai.data.connection import initialize_schema
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.data.repositories.daily_prices import get_latest_date, upsert_daily_prices
from tradingai.data.repositories.stocks import list_tickers
from tradingai.providers.registry import get_provider


def _ohlcv_rows(df) -> list[dict]:
    rows: list[dict] = []
    for timestamp, row in df.iterrows():
        rows.append(
            {
                "date": date.fromisoformat(timestamp.date().isoformat()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )
    return rows


def sync_daily_prices(tickers: list[str] | None = None, period: str = "2y") -> dict:
    initialize_schema()
    provider = get_provider()
    tickers = tickers or list_tickers()
    total = len(tickers)
    log_start("tradingai-sync-prices", total)
    synced = 0
    rows_written = 0
    errors: list[str] = []

    for index, ticker in enumerate(tickers, start=1):
        try:
            latest = get_latest_date(ticker)
            fetch_period = period
            if latest is not None:
                fetch_period = "6mo"
            df = provider.get_ohlcv(ticker, fetch_period, "1d")
            rows = _ohlcv_rows(df)
            if latest is not None:
                rows = [row for row in rows if row["date"] > latest]
            if not rows and latest is not None:
                synced += 1
                log_progress(index, total, ticker, f"up to date (latest {latest})")
                continue
            written = upsert_daily_prices(ticker, rows)
            rows_written += written
            synced += 1
            if written:
                latest_date = rows[-1]["date"]
                log_progress(index, total, ticker, f"{written} row(s), latest {latest_date}")
            else:
                log_progress(index, total, ticker, "no new rows")
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            log_error(index, total, ticker, str(exc))

    log_done("tradingai-sync-prices", synced, total, rows_written=rows_written, failed=len(errors))

    return {"synced": synced, "rows_written": rows_written, "errors": errors}


def main() -> None:
    import json

    result = sync_daily_prices()
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
