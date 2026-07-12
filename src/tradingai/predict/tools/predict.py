from __future__ import annotations

from tradingai.predict.adapters.news import fetch_news_signal
from tradingai.predict.adapters.regime import fetch_regime_signal
from tradingai.predict.adapters.technical import fetch_technical_signal
from tradingai.predict.evaluation.metrics import compute_evaluation_metrics
from tradingai.predict.evaluation.outcomes import evaluate_pending_outcomes
from tradingai.predict.fusion.scorer import explain_prediction_report, fuse_agent_signals
from tradingai.predict.storage.repository import (
    get_prediction_history,
    get_segment_calibration,
    get_track_record,
    save_prediction,
)


def get_agent_signals_tool(ticker: str) -> dict:
    signals = [
        fetch_technical_signal(ticker),
        fetch_news_signal(ticker),
        fetch_regime_signal(),
    ]
    return {
        "ticker": ticker.upper(),
        "agent_signals": [signal.model_dump() for signal in signals],
    }


def generate_prediction_tool(ticker: str) -> dict:
    signals = [
        fetch_technical_signal(ticker),
        fetch_news_signal(ticker),
        fetch_regime_signal(),
    ]
    report = fuse_agent_signals(ticker, signals)
    prediction_ids = save_prediction(report)
    payload = report.to_public_dict()
    payload["prediction_ids"] = prediction_ids
    return payload


def explain_prediction_tool(ticker: str) -> dict:
    signals = [
        fetch_technical_signal(ticker),
        fetch_news_signal(ticker),
        fetch_regime_signal(),
    ]
    report = fuse_agent_signals(ticker, signals)
    return {
        "ticker": ticker.upper(),
        "summary": explain_prediction_report(report),
        "fusion": report.fusion,
        "horizons": [horizon.model_dump() for horizon in report.horizons],
        "confidence": report.confidence,
        "reasons": report.reasons,
    }


def evaluate_predictions_tool(horizon_days: int | None = None) -> dict:
    if horizon_days is None:
        from tradingai.predict.config import get_prediction_horizons

        results = [
            evaluate_pending_outcomes(days) for days in get_prediction_horizons()
        ]
        return {"evaluation": results}
    return evaluate_pending_outcomes(horizon_days)


def get_evaluation_metrics_tool(horizon_days: int | None = None) -> dict:
    return compute_evaluation_metrics(horizon_days)


def get_prediction_history_tool(
    ticker: str | None = None,
    limit: int = 20,
) -> dict:
    history = get_prediction_history(ticker, limit=limit)
    return {
        "ticker": ticker.upper() if ticker else None,
        "count": len(history),
        "items": history,
    }


def get_prediction_track_record_tool(
    sector: str | None = None,
    prediction_type: str | None = None,
    catalyst_tag: str | None = None,
    regime: str | None = None,
    horizon_days: int | None = None,
) -> dict:
    return {
        "filters": {
            "sector": sector,
            "prediction_type": prediction_type,
            "catalyst_tag": catalyst_tag,
            "regime": regime,
            "horizon_days": horizon_days,
        },
        "track_record": get_track_record(
            sector=sector,
            prediction_type=prediction_type,
            catalyst_tag=catalyst_tag,
            regime=regime,
            horizon_days=horizon_days,
        ),
    }


def get_segment_calibration_tool(horizon_days: int | None = None) -> dict:
    return {
        "horizon_days": horizon_days,
        "segments": get_segment_calibration(horizon_days),
    }
