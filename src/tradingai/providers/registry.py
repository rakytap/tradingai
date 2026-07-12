from tradingai.config import get_data_provider_name
from tradingai.providers.alpaca_provider import AlpacaProvider
from tradingai.providers.base import MarketDataProvider
from tradingai.providers.yfinance_provider import YFinanceProvider

_PROVIDERS: dict[str, type[MarketDataProvider]] = {
    "yfinance": YFinanceProvider,
    "alpaca": AlpacaProvider,
}


def get_provider(name: str | None = None) -> MarketDataProvider:
    provider_name = (name or get_data_provider_name()).lower()
    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown data provider '{provider_name}'. Supported providers: {supported}."
        )
    return provider_cls()
