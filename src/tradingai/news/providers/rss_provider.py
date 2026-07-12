from __future__ import annotations

import time
from datetime import datetime, timezone

import feedparser

from tradingai.news.models import AnalystAction, NewsItem
from tradingai.news.providers.base import NewsProvider

DEFAULT_RSS_FEEDS: dict[str, str] = {
    "AAPL": "https://www.apple.com/newsroom/rss-feed.rss",
    "MSFT": "https://news.microsoft.com/feed/",
    "NVDA": "https://nvidianews.nvidia.com/releases.xml",
    "GOOGL": "https://blog.google/feed/",
    "AMZN": "https://www.aboutamazon.com/news/feed",
    "META": "https://about.fb.com/news/feed/",
    "TSLA": "https://www.tesla.com/blog/rss.xml",
}


def _parse_published(entry) -> datetime | None:
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if not published:
        return None
    return datetime(*published[:6], tzinfo=timezone.utc)


class RssNewsProvider(NewsProvider):
    name = "rss"

    def _resolve_feed(self, ticker: str, rss_url: str | None = None) -> str | None:
        if rss_url:
            return rss_url
        return DEFAULT_RSS_FEEDS.get(ticker.upper())

    def _fetch_feed_items(
        self, ticker: str, rss_url: str | None = None, count: int = 10
    ) -> list[NewsItem]:
        feed_url = self._resolve_feed(ticker, rss_url)
        if not feed_url:
            return []

        parsed = feedparser.parse(feed_url)
        items: list[NewsItem] = []
        for entry in parsed.entries[:count]:
            title = entry.get("title")
            if not title:
                continue
            summary = entry.get("summary") or entry.get("description")
            items.append(
                NewsItem(
                    title=title,
                    summary=summary,
                    url=entry.get("link"),
                    source=parsed.feed.get("title", "Company RSS"),
                    published_at=_parse_published(entry),
                    provider=self.name,
                )
            )
        return items

    def fetch_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return self._fetch_feed_items(ticker, count=count)

    def fetch_press_releases(self, ticker: str, count: int = 10) -> list[NewsItem]:
        return self._fetch_feed_items(ticker, count=count)

    def fetch_analyst_actions(self, ticker: str, count: int = 10) -> list[AnalystAction]:
        return []
