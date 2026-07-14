from __future__ import annotations

from tradingai.predict.models import AgentSignal
from tradingai.regime.analysis.scores import REGIME_SCORES


def adapt_regime_signal(payload: dict) -> AgentSignal:
    regime = str(payload.get("regime", "sideways"))
    directional = REGIME_SCORES.get(regime, 0.0)
    bullish_score = (directional + 1.0) / 2.0
    confidence = float(payload.get("confidence", 0.5))
    reasons = [signal.get("detail", "") for signal in payload.get("signals", [])[:2]]
    reasons = [reason for reason in reasons if reason]

    return AgentSignal(
        agent="regime",
        ticker=None,
        directional_score=round(directional, 4),
        bullish_score=round(bullish_score, 4),
        bearish_score=round(1.0 - bullish_score, 4),
        confidence=round(confidence, 4),
        reasons=reasons,
        raw=payload,
    )


def fetch_regime_signal() -> AgentSignal:
    from tradingai.regime.tools.regime import detect_market_regime

    payload = detect_market_regime()
    return adapt_regime_signal(payload)
