from __future__ import annotations

from tradingai.predict.adapters.news import fetch_news_signal
from tradingai.predict.adapters.ml_technical import fetch_technical_signal_with_backend
from tradingai.predict.adapters.regime import fetch_regime_signal
from tradingai.predict.config import (
    get_ml_horizon_days,
    get_ml_model_name,
    get_predict_backend,
)
from tradingai.predict.evaluation.metrics import compute_evaluation_metrics
from tradingai.predict.evaluation.outcomes import evaluate_pending_outcomes
from tradingai.predict.fusion.scorer import explain_prediction_report, fuse_agent_signals
from tradingai.predict.storage.repository import (
    get_prediction_history,
    get_segment_calibration,
    get_track_record,
    save_prediction,
)
from tradingai.predict.portfolio import build_paper_portfolio


def get_agent_signals_tool(ticker: str) -> dict:
    signals = [
        fetch_technical_signal_with_backend(ticker),
        fetch_news_signal(ticker),
        fetch_regime_signal(),
    ]
    return {
        "ticker": ticker.upper(),
        "agent_signals": [signal.model_dump() for signal in signals],
    }


def generate_prediction_tool(ticker: str) -> dict:
    signals = [
        fetch_technical_signal_with_backend(ticker),
        fetch_news_signal(ticker),
        fetch_regime_signal(),
    ]
    report = fuse_agent_signals(ticker, signals)
    if get_predict_backend() == "ml":
        report.fusion["backend"] = "ml"
        report.fusion["ml_model"] = get_ml_model_name()
        report.fusion["ml_horizon"] = get_ml_horizon_days()
    else:
        report.fusion["backend"] = "fusion"
    prediction_ids = save_prediction(report)
    payload = report.to_public_dict()
    payload["prediction_ids"] = prediction_ids
    return payload


def explain_prediction_tool(ticker: str) -> dict:
    signals = [
        fetch_technical_signal_with_backend(ticker),
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


def get_paper_portfolio_tool(
    *,
    positions: int = 10,
    cash_weight: float = 0.10,
    min_probability_up: float = 0.55,
    notional_usd: float = 100_000.0,
    horizon_days: int | None = None,
    optimizer: str = "score",
    lookback_days: int = 126,
    risk_aversion: float = 3.0,
    max_weight: float = 0.25,
) -> dict:
    return build_paper_portfolio(
        horizon_days=horizon_days,
        positions=positions,
        cash_weight=cash_weight,
        min_probability_up=min_probability_up,
        notional_usd=notional_usd,
        optimizer=optimizer,
        lookback_days=lookback_days,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )


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
