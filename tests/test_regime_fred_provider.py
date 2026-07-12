from unittest.mock import MagicMock, patch

import pandas as pd

from tradingai.regime.providers.fred_macro_provider import FredMacroProvider


@patch("tradingai.regime.providers.fred_macro_provider.get_fred_api_key", return_value="test-key")
@patch("fredapi.Fred")
def test_fred_provider_normalizes_series(mock_fred_cls, _mock_key):
    client = MagicMock()
    mock_fred_cls.return_value = client

    dates = pd.date_range("2024-01-01", periods=20, freq="ME")
    client.get_series.side_effect = [
        pd.Series([4.0 + i * 0.01 for i in range(20)], index=dates),
        pd.Series([3.8 + i * 0.02 for i in range(20)], index=dates),
        pd.Series([5.0 + i * 0.01 for i in range(20)], index=dates),
        pd.Series([2.2 + i * 0.01 for i in range(20)], index=dates),
        pd.Series([300 + i for i in range(20)], index=dates),
        pd.Series([4.0 + i * 0.05 for i in range(20)], index=dates),
    ]

    features = FredMacroProvider().fetch_features()

    assert features.treasury_10y is not None
    assert features.treasury_2y is not None
    assert features.yield_spread_2s10s_bps is not None
    assert features.yield_curve_inverted is not None
    assert features.fed_funds_rate is not None
    assert features.inflation_breakeven_10y is not None
    assert features.credit_spread_hy_oas is not None
    assert features.sources_used == ["fred"]
