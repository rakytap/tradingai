from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingai.providers.registry import get_provider
from tradingai.providers.yfinance_provider import YFinanceProvider, _normalize_ohlcv


def test_normalize_ohlcv_maps_expected_columns():
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    normalized = _normalize_ohlcv(raw)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.index.is_monotonic_increasing


def test_get_provider_defaults_to_yfinance():
    provider = get_provider("yfinance")
    assert isinstance(provider, YFinanceProvider)


def test_get_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown data provider"):
        get_provider("unknown-provider")


def test_yfinance_provider_normalizes_history():
    provider = YFinanceProvider()
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    with patch("tradingai.providers.yfinance_provider.yf.Ticker") as ticker_cls:
        ticker = MagicMock()
        ticker.history.return_value = raw
        ticker_cls.return_value = ticker

        df = provider.get_ohlcv("AAPL", "1mo", "1d")

    assert len(df) == 2
    assert "close" in df.columns


@pytest.mark.integration
def test_yfinance_provider_live_fetch():
    provider = YFinanceProvider()
    df = provider.get_ohlcv("AAPL", "1mo", "1d")
    assert not df.empty
    assert "close" in df.columns
