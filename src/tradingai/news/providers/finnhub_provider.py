from __future__ import annotations

from datetime import datetime, timezone

import httpx

from tradingai.news.config import get_finnhub_api_key
from tradingai.news.models import AnalystAction, NewsItem
from tradingai.news.providers.base import NewsProvider


class FinnhubProvider(NewsProvider):
    name = "finnhub"

    def _require_key(self) -> str:
        api_key = get_finnhub_api_key()
        if not api_key:
            raise ValueError(
                "FINNHUB_API_KEY is not configured. Add it to .env to enable the finnhub provider."
            )
        return api_key

    def fetch_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        api_key = self._require_key()
        params = {
            "symbol": ticker.upper(),
            "token": api_key,
        }
        with httpx.Client(timeout=20.0) as client:
            response = client.get("https://finnhub.io/api/v1/company-news", params=params)
            response.raise_for_status()
            payload = response.json()

        items: list[NewsItem] = []
        for article in payload[:count]:
            title = article.get("headline")
            if not title:
                continue
            published_at = None
            if article.get("datetime"):
                published_at = datetime.fromtimestamp(
                    article["datetime"], tz=timezone.utc
                )
            items.append(
                NewsItem(
                    title=title,
                    summary=article.get("summary"),
                    url=article.get("url"),
                    source=article.get("source", "Finnhub"),
                    published_at=published_at,
                    provider=self.name,
                )
            )
        return items

    def fetch_press_releases(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return [
            item
            for item in self.fetch_news(ticker, count=count * 2)
            if "press release" in item.title.lower()
        ][:count]

    def fetch_analyst_actions(self, ticker: str, count: int = 10) -> list[AnalystAction]:
        api_key = self._require_key()
        params = {
            "symbol": ticker.upper(),
            "token": api_key,
        }
        with httpx.Client(timeout=20.0) as client:
            response = client.get(
                "https://finnhub.io/api/v1/stock/recommendation", params=params
            )
            response.raise_for_status()
            payload = response.json()

        actions: list[AnalystAction] = []
        for row in payload[:count]:
            actions.append(
                AnalystAction(
                    firm="Finnhub aggregate",
                    action="recommendation_update",
                    from_grade=None,
                    to_grade=str(row.get("period")),
                    published_at=datetime.strptime(
                        str(row.get("period")), "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc)
                    if row.get("period")
                    else None,
                    provider=self.name,
                )
            )
        return actions
