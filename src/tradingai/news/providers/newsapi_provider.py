from __future__ import annotations

from datetime import datetime, timezone

import httpx

from tradingai.news.config import get_news_api_key
from tradingai.news.models import AnalystAction, NewsItem
from tradingai.news.providers.base import NewsProvider


class NewsApiProvider(NewsProvider):
    name = "newsapi"

    def _require_key(self) -> str:
        api_key = get_news_api_key()
        if not api_key:
            raise ValueError(
                "NEWS_API_KEY is not configured. Add it to .env to enable the newsapi provider."
            )
        return api_key

    def _search(self, query: str, count: int = 10) -> list[NewsItem]:
        api_key = self._require_key()
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": count,
            "apiKey": api_key,
        }
        with httpx.Client(timeout=20.0) as client:
            response = client.get("https://newsapi.org/v2/everything", params=params)
            response.raise_for_status()
            payload = response.json()

        items: list[NewsItem] = []
        for article in payload.get("articles", []):
            title = article.get("title")
            if not title:
                continue
            published = article.get("publishedAt")
            published_at = None
            if published:
                published_at = datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            items.append(
                NewsItem(
                    title=title,
                    summary=article.get("description"),
                    url=article.get("url"),
                    source=article.get("source", {}).get("name", "NewsAPI"),
                    published_at=published_at,
                    provider=self.name,
                )
            )
        return items

    def fetch_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return self._search(f"{ticker.upper()} stock", count=count)

    def fetch_press_releases(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return self._search(f"{ticker.upper()} press release", count=count)

    def fetch_analyst_actions(self, ticker: str, count: int = 10) -> list[AnalystAction]:
        return []
