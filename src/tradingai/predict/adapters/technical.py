from __future__ import annotations

from tradingai.predict.models import AgentSignal

BULLISH_TERMS = {
    "bullish",
    "bullish_crossover",
    "oversold",
    "strong_positive",
    "outperforming",
    "unusual_high",
}
BEARISH_TERMS = {
    "bearish",
    "bearish_crossover",
    "overbought",
    "strong_negative",
    "underperforming",
}


def _directional_from_signals(signals: list[dict]) -> tuple[float, list[str]]:
    bullish = 0
    bearish = 0
    reasons: list[str] = []

    for signal in signals:
        label = signal.get("signal", "")
        if label in BULLISH_TERMS:
            bullish += 1
            reasons.append(signal.get("detail", ""))
        elif label in BEARISH_TERMS:
            bearish += 1
            reasons.append(signal.get("detail", ""))

    total = bullish + bearish
    if total == 0:
        directional = 0.0
    else:
        directional = max(-1.0, min(1.0, (bullish - bearish) / total))

    return directional, [reason for reason in reasons[:3] if reason]


def adapt_technical_signal(ticker: str, payload: dict) -> AgentSignal:
    signals = payload.get("signals", [])
    directional, reasons = _directional_from_signals(signals)
    bullish_score = (directional + 1.0) / 2.0
    confidence = 0.4 + 0.4 * abs(directional)

    return AgentSignal(
        agent="technical",
        ticker=ticker.upper(),
        directional_score=round(directional, 4),
        bullish_score=round(bullish_score, 4),
        bearish_score=round(1.0 - bullish_score, 4),
        confidence=round(confidence, 4),
        reasons=reasons,
        raw=payload,
    )


def fetch_technical_signal(ticker: str) -> AgentSignal:
    from tradingai.tools.technical import analyze_technical_setup_tool

    payload = analyze_technical_setup_tool(ticker)
    return adapt_technical_signal(ticker, payload)
