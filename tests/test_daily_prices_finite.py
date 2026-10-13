from __future__ import annotations

import math
from datetime import date
from unittest.mock import patch

from tradingai.data.repositories.daily_prices import get_forward_return, is_finite_number, upsert_daily_prices


def test_is_finite_number():
    assert is_finite_number(1.5)
    assert not is_finite_number(float("nan"))
    assert not is_finite_number(float("inf"))
    assert not is_finite_number("x")


def test_upsert_daily_prices_skips_nan_rows():
    rows = [
        {
            "date": date(2026, 1, 2),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": float("nan"),
            "volume": 10,
        },
        {
            "date": date(2026, 1, 3),
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "volume": 10,
        },
    ]
    with patch("tradingai.data.repositories.daily_prices.get_connection") as mock_conn:
        conn = mock_conn.return_value.__enter__.return_value
        cur = conn.cursor.return_value.__enter__.return_value
        written = upsert_daily_prices("AAA", rows)
    assert written == 1
    args = cur.executemany.call_args[0][1]
    assert len(args) == 1
    assert args[0][5] == 1.05


def test_get_forward_return_skips_nan_closes():
    start = date(2026, 7, 23)
    db_rows = [
        (date(2026, 7, 23), 100.0),
        (date(2026, 7, 24), 101.0),
        (date(2026, 7, 27), float("nan")),
        (date(2026, 7, 28), 102.0),
        (date(2026, 7, 29), 103.0),
    ]
    with patch("tradingai.data.repositories.daily_prices.get_connection") as mock_conn:
        conn = mock_conn.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = db_rows
        ret = get_forward_return("AAPL", start, 3)
    # finite sessions: 100 -> 103 over 3 steps
    assert ret is not None
    assert math.isclose(ret, 3.0)
