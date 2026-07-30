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

    related_candidates = payload.get("related_candidates") or []
    l3_meta = None
    from tradingai.news.config import get_l3_news_spill_alpha, is_l3_news_spill_enabled

    if is_l3_news_spill_enabled() and related_candidates:
        from tradingai.news.analysis.l3_spillover import (
            blend_l3_news_score,
            load_neighbor_news_scores,
        )

        neighbor_scores = load_neighbor_news_scores(related_candidates)
        if neighbor_scores:
            directional, l3_meta = blend_l3_news_score(
                directional,
                related_candidates,
                neighbor_scores,
                get_l3_news_spill_alpha(),
            )
            bullish_score = (directional + 1.0) / 2.0

    bearish_score = 1.0 - bullish_score
    raw = {
        **payload,
        "sentiment": sentiment,
        "event_score": event_score,
        "events": payload.get("events") or [],
        "propagated_sectors": payload.get("propagated_sectors") or {},
        "related_candidates": related_candidates,
        "news_score_source": "events" if event_score is not None else "sentiment",
    }
    if l3_meta is not None:
        raw["l3_news_spill"] = l3_meta

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
