from tradingai.providers.base import MarketDataProvider


class AlpacaProvider(MarketDataProvider):
    """Stub for future Alpaca market data integration."""

    def get_ohlcv(self, symbol: str, period: str, interval: str):
        raise NotImplementedError(
            "AlpacaProvider is not implemented yet. "
            "Set TRADINGAI_DATA_PROVIDER=yfinance or implement Alpaca integration."
        )

    def get_quote(self, symbol: str):
        raise NotImplementedError(
            "AlpacaProvider is not implemented yet. "
            "Set TRADINGAI_DATA_PROVIDER=yfinance or implement Alpaca integration."
        )
