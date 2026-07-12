from __future__ import annotations

from datetime import date

from tradingai.data.connection import initialize_schema
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
    synced = 0
    rows_written = 0
    errors: list[str] = []

    for ticker in tickers:
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
                continue
            rows_written += upsert_daily_prices(ticker, rows)
            synced += 1
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    return {"synced": synced, "rows_written": rows_written, "errors": errors}


def main() -> None:
    import json

    print(json.dumps(sync_daily_prices(), indent=2))


if __name__ == "__main__":
    main()
