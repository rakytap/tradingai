from __future__ import annotations

import re
from datetime import datetime, timezone

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.news_events import upsert_news_event
from tradingai.data.repositories.stocks import upsert_stock
from tradingai.news.analysis.catalyst_extractor import CATALYST_PATTERNS
from tradingai.news.tools.news import analyze_news_sentiment


def _impact_from_headline(headline: str) -> str | None:
    lowered = headline.lower()
    for label, expressions in CATALYST_PATTERNS.items():
        if any(re.search(expr, lowered) for expr in expressions):
            return label
    return None


def sync_news_events(tickers: list[str] | None = None, count: int = 10) -> dict:
    initialize_schema()
    tickers = tickers or list(get_sync_tickers())
    synced = 0
    events_written = 0
    errors: list[str] = []

    for ticker in tickers:
        try:
            upsert_stock(ticker)
            payload = analyze_news_sentiment(ticker, count=count)
            sentiment = payload.get("sentiment")
            for headline in payload.get("catalysts", []):
                upsert_news_event(
                    ticker,
                    timestamp=datetime.now(timezone.utc),
                    headline=str(headline),
                    sentiment=sentiment,
                    impact=_impact_from_headline(str(headline)),
                    source="tradingai-news",
                    raw_json=payload,
                )
                events_written += 1
            synced += 1
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    return {"synced": synced, "events_written": events_written, "errors": errors}


def main() -> None:
    import json

    print(json.dumps(sync_news_events(), indent=2))


if __name__ == "__main__":
    main()
