from __future__ import annotations

import statistics

from tradingai.predict.config import (
    HORIZON_RETURN_MULTIPLIERS,
    get_fusion_weights,
    get_horizon_scales,
    get_prediction_horizons,
)
from tradingai.predict.models import AgentSignal, HorizonPrediction, PredictionReport


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = pow(2.718281828, -value)
        return 1.0 / (1.0 + z)
    z = pow(2.718281828, value)
    return z / (1.0 + z)


def _signal_by_agent(signals: list[AgentSignal]) -> dict[str, AgentSignal]:
    return {signal.agent: signal for signal in signals}


def compute_fusion_confidence(
    signals: list[AgentSignal],
    weights: dict[str, float],
) -> float:
    weighted_confidence = 0.0
    total_weight = 0.0
    for signal in signals:
        weight = weights.get(signal.agent, 0.0)
        weighted_confidence += signal.confidence * weight
        total_weight += weight

    base = weighted_confidence / total_weight if total_weight else 0.5
    directional_scores = [signal.directional_score for signal in signals]
    if len(directional_scores) > 1:
        agreement = 1.0 - (statistics.pstdev(directional_scores) / 2.0)
    else:
        agreement = 0.5

    confidence = 0.35 * base + 0.35 * agreement + 0.30
    return max(0.3, min(0.95, confidence))


def fuse_agent_signals(ticker: str, signals: list[AgentSignal]) -> PredictionReport:
    w_technical, w_news, w_regime = get_fusion_weights()
    by_agent = _signal_by_agent(signals)

    technical = by_agent.get("technical")
    news = by_agent.get("news")
    regime = by_agent.get("regime")

    technical_score = technical.directional_score if technical else 0.0
    news_score = news.directional_score if news else 0.0
    regime_score = regime.directional_score if regime else 0.0

    combined = (
        w_technical * technical_score
        + w_news * news_score
        + w_regime * regime_score
    )

    weights = {
        "technical": w_technical,
        "news": w_news,
        "regime": w_regime,
    }
    confidence = compute_fusion_confidence(signals, weights)

    horizons: list[HorizonPrediction] = []
    horizon_scales = get_horizon_scales()
    for horizon_days in get_prediction_horizons():
        scale = horizon_scales.get(horizon_days, 1.0)
        return_multiplier = HORIZON_RETURN_MULTIPLIERS.get(horizon_days, 1.0)
        horizons.append(
            HorizonPrediction(
                horizon_days=horizon_days,
                probability_up=round(_sigmoid(combined * scale), 4),
                expected_return_pct=round(combined * return_multiplier, 2),
            )
        )

    reasons = collect_reasons(signals)

    return PredictionReport(
        ticker=ticker.upper(),
        agent_signals=signals,
        fusion={
            "technical_score": round(technical_score, 4),
            "news_score": round(news_score, 4),
            "regime_score": round(regime_score, 4),
            "combined_score": round(combined, 4),
            "weights": weights,
        },
        horizons=horizons,
        confidence=round(confidence, 4),
        reasons=reasons,
    )


def collect_reasons(signals: list[AgentSignal], limit: int = 5) -> list[str]:
    reasons: list[str] = []
    for signal in signals:
        for reason in signal.reasons:
            if reason and reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= limit:
                return reasons
    return reasons


def explain_prediction_report(report: PredictionReport) -> str:
    fusion = report.fusion
    horizon_lines = [
        (
            f"{horizon.horizon_days} trading day(s): "
            f"{horizon.probability_up * 100:.1f}% up, "
            f"expected return {horizon.expected_return_pct:+.2f}%"
        )
        for horizon in report.horizons
    ]
    reason_text = "; ".join(report.reasons[:5])
    disclaimer = (
        "Probabilities are computed by a deterministic fusion function, not by an LLM. "
        "This is informational analysis only and not financial advice."
    )
    return (
        f"Prediction for {report.ticker}: combined score {fusion['combined_score']:.2f} "
        f"(technical {fusion['technical_score']:.2f}, news {fusion['news_score']:.2f}, "
        f"regime {fusion['regime_score']:.2f}) with confidence {report.confidence:.2f}. "
        f"Horizons: {' | '.join(horizon_lines)}. Reasons: {reason_text}. {disclaimer}"
    )
