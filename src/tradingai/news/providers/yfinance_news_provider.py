from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from tradingai.news.models import AnalystAction, NewsItem
from tradingai.news.providers.base import NewsProvider


def _parse_datetime(value) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except (TypeError, ValueError):
        return None


def _normalize_yfinance_news(items: list[dict], provider: str) -> list[NewsItem]:
    normalized: list[NewsItem] = []
    for item in items or []:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if not title:
            continue
        normalized.append(
            NewsItem(
                title=title,
                summary=content.get("summary") or content.get("description"),
                url=content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
                or item.get("link"),
                source=content.get("provider", {}).get("displayName")
                or item.get("publisher")
                or "Yahoo Finance",
                published_at=_parse_datetime(
                    content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
                ),
                provider=provider,
            )
        )
    return normalized


class YFinanceNewsProvider(NewsProvider):
    name = "yfinance"

    def fetch_news(self, ticker: str, count: int = 10) -> list[NewsItem]:
        stock = yf.Ticker(ticker.upper())
        return _normalize_yfinance_news(stock.get_news(count=count, tab="news"), self.name)

    def fetch_press_releases(self, ticker: str, count: int = 10) -> list[NewsItem]:
        stock = yf.Ticker(ticker.upper())
        return _normalize_yfinance_news(
            stock.get_news(count=count, tab="press releases"), self.name
        )

    def fetch_analyst_actions(self, ticker: str, count: int = 10) -> list[AnalystAction]:
        stock = yf.Ticker(ticker.upper())
        actions: list[AnalystAction] = []
        try:
            frame = stock.upgrades_downgrades
        except Exception:
            return actions

        if frame is None or getattr(frame, "empty", True):
            return actions

        records = frame.reset_index().tail(count)
        for _, row in records.iterrows():
            action = str(row.get("Action") or row.get("Grade") or "update")
            actions.append(
                AnalystAction(
                    firm=str(row.get("Firm") or row.get("Brokerage") or "Unknown"),
                    action=action,
                    from_grade=str(row.get("FromGrade")) if pd.notna(row.get("FromGrade")) else None,
                    to_grade=str(row.get("ToGrade")) if pd.notna(row.get("ToGrade")) else None,
                    published_at=_parse_datetime(row.get("Date") or row.get("index")),
                    provider=self.name,
                )
            )
        return actions
