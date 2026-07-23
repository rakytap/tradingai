from __future__ import annotations

from tradingai.news.analysis.social_noise_filter import filter_social_posts
from tradingai.news.analysis.sentiment_scorer import score_social_posts
from tradingai.news.analysis.synthesizer import synthesize_sentiment_report
from tradingai.news.models import AnalystAction, EarningsMaterial, FilingItem, NewsItem, SocialPost
from tradingai.news.providers.registry import (
    get_enabled_news_providers,
    get_enabled_social_providers,
    get_filings_provider,
)
from tradingai.news.providers.rss_provider import RssNewsProvider


def _dedupe_news(items: list[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    deduped: list[NewsItem] = []
    for item in items:
        key = item.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _collect_from_providers(
    ticker: str,
    method_name: str,
    *,
    count: int = 10,
    rss_url: str | None = None,
) -> tuple[list, list[str], list[str]]:
    collected = []
    sources_with_data: list[str] = []
    sources_checked: list[str] = []
    for provider in get_enabled_news_providers():
        try:
            if isinstance(provider, RssNewsProvider):
                if method_name == "fetch_news":
                    items = provider._fetch_feed_items(
                        ticker, rss_url=rss_url, count=count
                    )
                elif method_name == "fetch_press_releases":
                    items = provider._fetch_feed_items(
                        ticker, rss_url=rss_url, count=count
                    )
                elif method_name == "fetch_analyst_actions":
                    items = provider.fetch_analyst_actions(ticker, count=count)
                else:
                    continue
            elif method_name == "fetch_news":
                items = provider.fetch_news(ticker, count=count)
            elif method_name == "fetch_press_releases":
                items = provider.fetch_press_releases(ticker, count=count)
            elif method_name == "fetch_analyst_actions":
                items = provider.fetch_analyst_actions(ticker, count=count)
            else:
                continue
        except ValueError:
            continue
        except Exception:
            sources_checked.append(provider.name)
            continue
        sources_checked.append(provider.name)
        if items:
            sources_with_data.append(provider.name)
            collected.extend(items)
    return collected, sources_with_data, sources_checked


def _collect_social_posts(
    ticker: str, count: int = 15
) -> tuple[list[SocialPost], list[str], list[str], int]:
    collected: list[SocialPost] = []
    sources_with_data: list[str] = []
    sources_checked: list[str] = []
    for provider in get_enabled_social_providers():
        try:
            posts = provider.fetch_posts(ticker, count=count)
        except ValueError:
            continue
        except Exception:
            sources_checked.append(provider.name)
            continue
        sources_checked.append(provider.name)
        if posts:
            sources_with_data.append(provider.name)
            collected.extend(posts)
    filtered, noise_removed = filter_social_posts(collected, ticker)
    return filtered, sources_with_data, sources_checked, noise_removed


def fetch_social_sentiment(ticker: str, count: int = 15) -> dict:
    """Fetch filtered Reddit/X posts and a social-only sentiment score."""
    posts, sources, _, noise_removed = _collect_social_posts(ticker, count=count)
    social_score, _ = score_social_posts(posts)
    return {
        "ticker": ticker.upper(),
        "count": len(posts),
        "noise_filtered": noise_removed,
        "social_sentiment": round(social_score, 2) if social_score is not None else None,
        "sources": sorted(set(sources)),
        "items": [item.model_dump(mode="json") for item in posts[:count]],
    }


def fetch_company_news(
    ticker: str,
    count: int = 10,
    rss_url: str | None = None,
) -> dict:
    items, sources, _ = _collect_from_providers(
        ticker, "fetch_news", count=count, rss_url=rss_url
    )
    deduped = _dedupe_news(items)
    return {
        "ticker": ticker.upper(),
        "count": len(deduped),
        "sources": sorted(set(sources)),
        "items": [item.model_dump(mode="json") for item in deduped[:count]],
    }


def fetch_press_releases(
    ticker: str,
    count: int = 10,
    rss_url: str | None = None,
) -> dict:
    items, sources, _ = _collect_from_providers(
        ticker, "fetch_press_releases", count=count, rss_url=rss_url
    )
    deduped = _dedupe_news(items)
    return {
        "ticker": ticker.upper(),
        "count": len(deduped),
        "sources": sorted(set(sources)),
        "items": [item.model_dump(mode="json") for item in deduped[:count]],
    }


def fetch_sec_filings(
    ticker: str,
    forms: list[str] | None = None,
    count: int = 5,
) -> dict:
    provider = get_filings_provider("sec_edgar")
    try:
        filings = provider.fetch_filings(ticker, forms=forms, count=count)
    except ValueError as exc:
        return {
            "ticker": ticker.upper(),
            "count": 0,
            "sources": [],
            "items": [],
            "error": str(exc),
        }
    return {
        "ticker": ticker.upper(),
        "count": len(filings),
        "sources": ["sec_edgar"],
        "items": [item.model_dump(mode="json") for item in filings],
    }


def fetch_earnings_materials(ticker: str) -> dict:
    provider = get_filings_provider("sec_edgar")
    try:
        materials = provider.fetch_earnings_materials(ticker)
    except ValueError as exc:
        return {
            "ticker": ticker.upper(),
            "sources": [],
            "materials": None,
            "error": str(exc),
        }
    return {
        "ticker": ticker.upper(),
        "sources": ["sec_edgar"] if materials else [],
        "materials": materials.model_dump(mode="json") if materials else None,
    }


def fetch_analyst_actions(ticker: str, count: int = 10) -> dict:
    actions, sources, _ = _collect_from_providers(
        ticker, "fetch_analyst_actions", count=count
    )
    return {
        "ticker": ticker.upper(),
        "count": len(actions),
        "sources": sorted(set(sources)),
        "items": [item.model_dump(mode="json") for item in actions[:count]],
    }


def analyze_news_sentiment(
    ticker: str,
    count: int = 10,
    rss_url: str | None = None,
) -> dict:
    news_items, news_sources, news_checked = _collect_from_providers(
        ticker, "fetch_news", count=count, rss_url=rss_url
    )
    press_items, press_sources, press_checked = _collect_from_providers(
        ticker, "fetch_press_releases", count=count, rss_url=rss_url
    )
    analyst_actions, analyst_sources, analyst_checked = _collect_from_providers(
        ticker, "fetch_analyst_actions", count=count
    )
    social_posts, social_sources, social_checked, noise_removed = _collect_social_posts(
        ticker, count=count
    )

    filings: list[FilingItem] = []
    filing_sources: list[str] = []
    filing_checked: list[str] = []
    earnings: EarningsMaterial | None = None

    try:
        filings_provider = get_filings_provider("sec_edgar")
        filing_checked.append("sec_edgar")
        filings = filings_provider.fetch_filings(ticker, count=3)
        earnings = filings_provider.fetch_earnings_materials(ticker)
        if filings or earnings:
            filing_sources.append("sec_edgar")
    except ValueError:
        pass
    except Exception:
        filing_checked.append("sec_edgar")

    sources_with_data = sorted(
        set(news_sources + press_sources + analyst_sources + filing_sources + social_sources)
    )
    sources_checked = sorted(
        set(
            news_checked
            + press_checked
            + analyst_checked
            + social_checked
            + filing_checked
        )
    )

    report = synthesize_sentiment_report(
        ticker=ticker,
        news=_dedupe_news(news_items),
        press_releases=_dedupe_news(press_items),
        filings=filings,
        analyst_actions=analyst_actions,
        earnings=earnings,
        social_posts=social_posts,
        sources_used=sources_with_data,
    )

    social_score, _ = score_social_posts(social_posts)

    return {
        **report.to_public_dict(),
        "headline_count": len(_dedupe_news(news_items)),
        "press_release_count": len(_dedupe_news(press_items)),
        "analyst_action_count": len(analyst_actions),
        "filing_count": len(filings),
        "social_post_count": len(social_posts),
        "social_noise_filtered": noise_removed,
        "social_sentiment": round(social_score, 2) if social_score is not None else None,
        "source_stats": {
            "checked": sources_checked,
            "with_data": sources_with_data,
        },
    }


def get_news_events(ticker: str, limit: int = 20) -> dict:
    """Return L2 typed news-graph events for a ticker (live extract + DB history)."""
    from tradingai.data.connection import initialize_schema
    from tradingai.data.repositories.news_graph_events import (
        get_news_graph_events,
        save_news_graph_events,
    )
    from tradingai.data.repositories.stocks import get_stock

    live = analyze_news_sentiment(ticker)
    events = live.get("events") or []
    try:
        initialize_schema()
        save_news_graph_events(ticker, events=events)
        stored = get_news_graph_events(ticker, limit=limit)
    except Exception:
        stored = []

    stock = None
    try:
        stock = get_stock(ticker)
    except Exception:
        stock = None
    sector = stock.get("sector") if stock else None
    return {
        "ticker": ticker.upper(),
        "sector": sector,
        "event_score": live.get("event_score"),
        "event_confidence": live.get("event_confidence"),
        "propagated_sectors": live.get("propagated_sectors") or {},
        "event_reasons": live.get("event_reasons") or [],
        "live_events": events,
        "stored_events": stored,
        "sentiment": live.get("sentiment"),
        "news_score_source": "events" if live.get("event_score") is not None else "sentiment",
    }
