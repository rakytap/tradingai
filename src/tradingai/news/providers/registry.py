from tradingai.news.config import get_news_providers, get_social_providers
from tradingai.news.providers.base import FilingsProvider, NewsProvider, SocialProvider
from tradingai.news.providers.finnhub_provider import FinnhubProvider
from tradingai.news.providers.newsapi_provider import NewsApiProvider
from tradingai.news.providers.reddit_provider import RedditProvider
from tradingai.news.providers.rss_provider import RssNewsProvider
from tradingai.news.providers.sec_edgar_provider import SecEdgarProvider
from tradingai.news.providers.x_provider import XProvider
from tradingai.news.providers.yfinance_news_provider import YFinanceNewsProvider

_NEWS_PROVIDERS: dict[str, type[NewsProvider]] = {
    "yfinance": YFinanceNewsProvider,
    "rss": RssNewsProvider,
    "newsapi": NewsApiProvider,
    "finnhub": FinnhubProvider,
}

_SOCIAL_PROVIDERS: dict[str, type[SocialProvider]] = {
    "reddit": RedditProvider,
    "x": XProvider,
}

_FILINGS_PROVIDERS: dict[str, type[FilingsProvider]] = {
    "sec_edgar": SecEdgarProvider,
}


def get_enabled_news_providers(names: list[str] | None = None) -> list[NewsProvider]:
    provider_names = names or get_news_providers()
    providers: list[NewsProvider] = []
    for name in provider_names:
        provider_cls = _NEWS_PROVIDERS.get(name)
        if provider_cls is None:
            continue
        providers.append(provider_cls())
    return providers


def get_enabled_social_providers(names: list[str] | None = None) -> list[SocialProvider]:
    provider_names = names or get_social_providers()
    providers: list[SocialProvider] = []
    for name in provider_names:
        provider_cls = _SOCIAL_PROVIDERS.get(name)
        if provider_cls is None:
            continue
        providers.append(provider_cls())
    return providers


def get_filings_provider(name: str = "sec_edgar") -> FilingsProvider:
    provider_cls = _FILINGS_PROVIDERS.get(name)
    if provider_cls is None:
        supported = ", ".join(sorted(_FILINGS_PROVIDERS))
        raise ValueError(f"Unknown filings provider '{name}'. Supported: {supported}.")
    return provider_cls()
