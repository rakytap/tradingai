from __future__ import annotations

from unittest.mock import patch

from tradingai.data.etf_symbols import is_etf
from tradingai.data.stock_metadata import ETF_INDUSTRY, ETF_SECTOR, resolve_stock_metadata


def test_is_etf_recognizes_known_symbols():
    assert is_etf("BIL")
    assert is_etf("spy")
    assert not is_etf("AAPL")


def test_resolve_stock_metadata_for_etf_skips_yfinance_info():
    with patch("tradingai.data.stock_metadata.yf.Ticker") as ticker_cls:
        metadata = resolve_stock_metadata("BIL")

    ticker_cls.assert_not_called()
    assert metadata.sector == ETF_SECTOR
    assert metadata.industry == ETF_INDUSTRY
    assert metadata.market_cap is None


def test_resolve_stock_metadata_for_stock_uses_yfinance_info():
    mock_info = {
        "sector": "Technology",
        "industry": "Semiconductors",
        "marketCap": 1_000_000,
    }
    with patch("tradingai.data.stock_metadata.yf.Ticker") as ticker_cls:
        ticker_cls.return_value.info = mock_info
        metadata = resolve_stock_metadata("NVDA")

    assert metadata.sector == "Technology"
    assert metadata.industry == "Semiconductors"
    assert metadata.market_cap == 1_000_000
