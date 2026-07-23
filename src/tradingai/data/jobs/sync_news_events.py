from __future__ import annotations

import re
from datetime import datetime, timezone

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.data.repositories.news_events import upsert_news_event
from tradingai.data.repositories.news_graph_events import save_news_graph_events
from tradingai.data.repositories.stocks import upsert_stock
from tradingai.news.analysis.catalyst_extractor import CATALYST_PATTERNS
from tradingai.news.config import get_news_providers, get_social_providers
from tradingai.news.tools.news import analyze_news_sentiment


def _format_sources(sources: list[str]) -> str:
    return ", ".join(sources) if sources else "none"


def _configured_source_names() -> list[str]:
    return sorted(set(get_news_providers()) | set(get_social_providers()) | {"sec_edgar"})


def _news_progress_message(
    *,
    ticker_events: int,
    sentiment: float | None,
    source_stats: dict,
) -> str:
    checked = source_stats.get("checked", [])
    with_data = source_stats.get("with_data", [])
    db_sources = with_data if ticker_events > 0 else []
    sentiment_label = (
        f"sentiment {sentiment:.2f}" if sentiment is not None else "no sentiment"
    )
    return (
        f"{ticker_events} event(s), {sentiment_label} | "
        f"checked: {_format_sources(checked)} | "
        f"with data: {_format_sources(with_data)} | "
        f"db: {_format_sources(db_sources)}"
    )

def _impact_from_headline(headline: str) -> str | None:
    lowered = headline.lower()
    for label, expressions in CATALYST_PATTERNS.items():
        if any(re.search(expr, lowered) for expr in expressions):
            return label
    return None


def sync_news_events(tickers: list[str] | None = None, count: int = 10) -> dict:
    initialize_schema()
    tickers = tickers or list(get_sync_tickers())
    total = len(tickers)
    log_start("tradingai-sync-news", total)
    print(
        f"tradingai-sync-news: configured sources: {_format_sources(_configured_source_names())}",
        flush=True,
    )
    synced = 0
    events_written = 0
    errors: list[str] = []
    source_summary: dict[str, int] = {}
    db_source_summary: dict[str, int] = {}

    for index, ticker in enumerate(tickers, start=1):
        try:
            upsert_stock(ticker)
            payload = analyze_news_sentiment(ticker, count=count)
            sentiment = payload.get("sentiment")
            source_stats = payload.get("source_stats", {})
            ticker_events = 0
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
                ticker_events += 1
            try:
                save_news_graph_events(ticker, events=payload.get("events") or [])
            except Exception:
                pass
            synced += 1
            for source in source_stats.get("checked", []):
                source_summary[source] = source_summary.get(source, 0) + 1
            if ticker_events > 0:
                for source in source_stats.get("with_data", []):
                    db_source_summary[source] = db_source_summary.get(source, 0) + 1
            log_progress(
                index,
                total,
                ticker,
                _news_progress_message(
                    ticker_events=ticker_events,
                    sentiment=sentiment,
                    source_stats=source_stats,
                ),
            )
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            log_error(index, total, ticker, str(exc))

    log_done(
        "tradingai-sync-news",
        synced,
        total,
        events_written=events_written,
        failed=len(errors),
    )
    if source_summary:
        checked_totals = ", ".join(
            f"{name}={count}" for name, count in sorted(source_summary.items())
        )
        print(f"tradingai-sync-news: source check totals: {checked_totals}", flush=True)
    if db_source_summary:
        db_totals = ", ".join(
            f"{name}={count}" for name, count in sorted(db_source_summary.items())
        )
        print(f"tradingai-sync-news: source db totals: {db_totals}", flush=True)

    return {
        "synced": synced,
        "events_written": events_written,
        "errors": errors,
        "source_check_totals": source_summary,
        "source_db_totals": db_source_summary,
    }


def main() -> None:
    import json

    result = sync_news_events()
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
