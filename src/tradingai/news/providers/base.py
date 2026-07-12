from __future__ import annotations

from abc import ABC, abstractmethod

from tradingai.news.models import AnalystAction, EarningsMaterial, FilingItem, NewsItem, SocialPost


class NewsProvider(ABC):
    name: str

    @abstractmethod
    def fetch_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        """Fetch recent news headlines for a ticker."""

    @abstractmethod
    def fetch_press_releases(self, ticker: str, count: int = 10) -> list[NewsItem]:
        """Fetch press releases / IR announcements for a ticker."""

    @abstractmethod
    def fetch_analyst_actions(self, ticker: str, count: int = 10) -> list[AnalystAction]:
        """Fetch analyst upgrades/downgrades for a ticker."""


class FilingsProvider(ABC):
    name: str

    @abstractmethod
    def fetch_filings(
        self, ticker: str, forms: list[str] | None = None, count: int = 5
    ) -> list[FilingItem]:
        """Fetch recent SEC filings for a ticker."""

    @abstractmethod
    def fetch_earnings_materials(self, ticker: str) -> EarningsMaterial | None:
        """Fetch latest earnings press release and transcript when available."""


class SocialProvider(ABC):
    name: str

    @abstractmethod
    def fetch_posts(self, ticker: str, count: int = 15) -> list[SocialPost]:
        """Fetch social media posts mentioning a ticker."""
