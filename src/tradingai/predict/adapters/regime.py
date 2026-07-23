from __future__ import annotations

from tradingai.predict.models import AgentSignal
from tradingai.regime.analysis.scores import REGIME_SCORES
from tradingai.regime.sector_proxies import normalize_sector_label, resolve_regime_proxy_for_ticker

STANCE_SCORES = {
    "bullish": 0.80,
    "neutral": 0.00,
    "bearish": -0.75,
}


def adapt_regime_signal(payload: dict) -> AgentSignal:
    regime = str(payload.get("regime", "sideways"))
    directional = float(payload.get("directional_score", REGIME_SCORES.get(regime, 0.0)))
    bullish_score = (directional + 1.0) / 2.0
    confidence = float(payload.get("confidence", 0.5))
    reasons = [reason for reason in payload.get("reasons", []) if reason]

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


def fetch_regime_signal(ticker: str) -> AgentSignal:
    from tradingai.regime.tools.regime import load_or_build_regime_state

    state = load_or_build_regime_state()
    sector, proxy = resolve_regime_proxy_for_ticker(ticker)
    canonical = normalize_sector_label(sector) or sector
    stance = state.stance_for_sector(canonical) if canonical else None

    if stance is not None:
        regime_label = stance
        directional = STANCE_SCORES.get(stance, 0.0)
        reasons = [
            f"Sector board {canonical}: {stance} (proxy {proxy})",
            f"Market risk={state.market.risk} vol={state.market.vol} "
            f"equity_trend={state.market.equity_trend}",
        ]
    else:
        regime_label = f"market:{state.market.equity_trend}"
        directional = REGIME_SCORES.get(state.market_regime, 0.0)
        reasons = [
            f"No sector board entry; using market equity_trend={state.market.equity_trend}",
            f"Market risk={state.market.risk} vol={state.market.vol}",
        ]

    payload = {
        "regime": regime_label,
        "directional_score": directional,
        "confidence": state.confidence,
        "reasons": reasons,
        "sector": canonical,
        "proxy": proxy,
        "stance": stance,
        "market_regime": state.market_regime,
        "regime_state": state.to_public_dict(),
    }
    return adapt_regime_signal(payload)
