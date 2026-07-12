from datetime import date
from unittest.mock import patch

import pytest

from tradingai.data.repositories.daily_prices import get_forward_return


def test_get_forward_return_computes_percentage():
    rows = [
        (date(2026, 1, 2), 100.0),
        (date(2026, 1, 3), 101.0),
        (date(2026, 1, 6), 102.0),
        (date(2026, 1, 7), 103.0),
        (date(2026, 1, 8), 105.0),
        (date(2026, 1, 9), 106.0),
    ]

    mock_conn = type("Conn", (), {})()
    mock_conn.execute = lambda *args, **kwargs: type(
        "Result",
        (),
        {"fetchall": lambda self: rows},
    )()

    with patch("tradingai.data.repositories.daily_prices.get_connection") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_conn
        result = get_forward_return("AAPL", date(2026, 1, 2), 4)

    assert result == pytest.approx(5.0)
