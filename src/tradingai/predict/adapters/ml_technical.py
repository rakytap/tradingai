from __future__ import annotations

import logging

from tradingai.predict.adapters.technical import adapt_technical_signal, fetch_technical_signal
from tradingai.predict.config import get_ml_horizon_days, get_ml_model_name, get_predict_backend
from tradingai.predict.models import AgentSignal

logger = logging.getLogger(__name__)


def _ml_reasons(payload: dict) -> list[str]:
    probability_up = payload.get("probability_up")
    model = payload.get("model")
    horizon_days = payload.get("horizon_days")
    return [
        (
            f"ML {model} model predicts {probability_up * 100:.1f}% probability up "
            f"over {horizon_days} trading day(s)."
        )
    ]


def adapt_ml_technical_signal(ticker: str, payload: dict) -> AgentSignal:
    directional = float(payload["directional_score"])
    probability_up = float(payload["probability_up"])
    bullish_score = (directional + 1.0) / 2.0
    confidence = 0.4 + 0.4 * abs(directional)

    return AgentSignal(
        agent="technical",
        ticker=ticker.upper(),
        directional_score=round(directional, 4),
        bullish_score=round(bullish_score, 4),
        bearish_score=round(1.0 - bullish_score, 4),
        confidence=round(confidence, 4),
        reasons=_ml_reasons(payload),
        raw=payload,
    )


def fetch_ml_technical_signal(ticker: str) -> AgentSignal:
    from tradingai.analysis.ml.inference import predict_direction

    payload = predict_direction(
        ticker,
        model_name=get_ml_model_name(),
        horizon_days=get_ml_horizon_days(),
    )
    return adapt_ml_technical_signal(ticker, payload)


def fetch_technical_signal_with_backend(ticker: str) -> AgentSignal:
    if get_predict_backend() != "ml":
        return fetch_technical_signal(ticker)

    try:
        return fetch_ml_technical_signal(ticker)
    except Exception as exc:
        logger.warning(
            "ML technical signal unavailable for %s (%s); falling back to rule-based adapter.",
            ticker,
            exc,
        )
        return fetch_technical_signal(ticker)
