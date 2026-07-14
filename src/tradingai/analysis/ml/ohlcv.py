from __future__ import annotations

import pandas as pd

from tradingai.data.connection import get_connection


def load_ticker_ohlcv(
    ticker: str,
    *,
    database_url: str | None = None,
) -> pd.DataFrame:
    with get_connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM daily_prices
            WHERE ticker = %s
            ORDER BY date ASC
            """,
            (ticker.upper(),),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    dates = [row[0] for row in rows]
    data = {
        "open": [float(row[1]) for row in rows],
        "high": [float(row[2]) for row in rows],
        "low": [float(row[3]) for row in rows],
        "close": [float(row[4]) for row in rows],
        "volume": [int(row[5]) for row in rows],
    }
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame(data, index=index)
