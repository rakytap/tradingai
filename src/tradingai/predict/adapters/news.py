from __future__ import annotations

from tradingai.predict.models import AgentSignal


def adapt_news_signal(ticker: str, payload: dict) -> AgentSignal:
    sentiment = float(payload.get("sentiment", 0.5))
    event_score = payload.get("event_score")
    event_confidence = payload.get("event_confidence")

    if event_score is not None:
        directional = max(-1.0, min(1.0, float(event_score)))
        bullish_score = (directional + 1.0) / 2.0
        confidence = float(event_confidence) if event_confidence is not None else float(
            payload.get("confidence", 0.5)
        )
        confidence = max(0.4, min(0.95, confidence))
        reasons = list(payload.get("event_reasons") or [])[:4]
        if not reasons:
            for event in (payload.get("events") or [])[:2]:
                reasons.append(str(event.get("event_type", "event")))
    else:
        # Fallback when no typed events matched.
        directional = max(-1.0, min(1.0, (sentiment - 0.5) * 2.0))
        bullish_score = sentiment
        confidence = float(payload.get("confidence", 0.5))
        reasons = []
        for catalyst in payload.get("catalysts", [])[:2]:
            reasons.append(str(catalyst))
        for risk in payload.get("risk_factors", [])[:1]:
            reasons.append(f"Risk: {risk}")

    bearish_score = 1.0 - bullish_score
    raw = {
        **payload,
        "sentiment": sentiment,
        "event_score": event_score,
        "events": payload.get("events") or [],
        "propagated_sectors": payload.get("propagated_sectors") or {},
        "news_score_source": "events" if event_score is not None else "sentiment",
    }

    return AgentSignal(
        agent="news",
        ticker=ticker.upper(),
        directional_score=round(directional, 4),
        bullish_score=round(bullish_score, 4),
        bearish_score=round(bearish_score, 4),
        confidence=round(confidence, 4),
        reasons=reasons,
        raw=raw,
    )


def fetch_news_signal(ticker: str) -> AgentSignal:
    from tradingai.news.tools.news import analyze_news_sentiment
    from tradingai.data.repositories.news_events import (
        get_latest_sentiment_snapshot,
        save_sentiment_snapshot,
    )
    from tradingai.data.repositories.news_graph_events import save_news_graph_events

    cached = get_latest_sentiment_snapshot(ticker, max_age_minutes=120)
    if cached is not None:
        cached = {**cached, "cache": {"hit": True, "max_age_minutes": 120}}
        return adapt_news_signal(ticker, cached)

    payload = analyze_news_sentiment(ticker)
    try:
        save_sentiment_snapshot(ticker, payload=payload)
        save_news_graph_events(ticker, events=payload.get("events") or [])
    except Exception:
        # Cache / graph write should never block predictions.
        pass
    payload = {**payload, "cache": {"hit": False, "max_age_minutes": 120}}
    return adapt_news_signal(ticker, payload)
