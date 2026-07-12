from __future__ import annotations

from tradingai.predict.models import AgentSignal


def adapt_news_signal(ticker: str, payload: dict) -> AgentSignal:
    sentiment = float(payload.get("sentiment", 0.5))
    directional = max(-1.0, min(1.0, (sentiment - 0.5) * 2.0))
    bullish_score = sentiment
    bearish_score = 1.0 - sentiment
    confidence = float(payload.get("confidence", 0.5))

    reasons: list[str] = []
    for catalyst in payload.get("catalysts", [])[:2]:
        reasons.append(str(catalyst))
    for risk in payload.get("risk_factors", [])[:1]:
        reasons.append(f"Risk: {risk}")

    return AgentSignal(
        agent="news",
        ticker=ticker.upper(),
        directional_score=round(directional, 4),
        bullish_score=round(bullish_score, 4),
        bearish_score=round(bearish_score, 4),
        confidence=round(confidence, 4),
        reasons=reasons,
        raw=payload,
    )


def fetch_news_signal(ticker: str) -> AgentSignal:
    from tradingai.news.tools.news import analyze_news_sentiment

    payload = analyze_news_sentiment(ticker)
    return adapt_news_signal(ticker, payload)
