from __future__ import annotations

from datetime import datetime, timezone

from tradingai.data.repositories import predictions as predictions_repo
from tradingai.data.repositories.stocks import get_stock, upsert_stock
from tradingai.data.stock_metadata import resolve_stock_metadata
from tradingai.predict.config import get_ml_model_name, get_predict_backend
from tradingai.predict.models import PredictionReport
from tradingai.predict.tagging.prediction_tagger import build_prediction_metadata


def ensure_stock(ticker: str) -> dict | None:
    stock = get_stock(ticker)
    if stock is not None:
        return stock
    try:
        metadata = resolve_stock_metadata(ticker)
        upsert_stock(
            ticker,
            sector=metadata.sector,
            industry=metadata.industry,
            market_cap=metadata.market_cap,
        )
    except Exception:
        upsert_stock(ticker)
    return get_stock(ticker)


def save_prediction(report: PredictionReport) -> list[int]:
    stock = ensure_stock(report.ticker)
    sector = stock.get("sector") if stock else None
    metadata = build_prediction_metadata(report, sector=sector)
    report.metadata = metadata

    agent_payload = {signal.agent: signal.raw for signal in report.agent_signals}
    timestamp = report.as_of.astimezone(timezone.utc)
    model_name = None
    model_version = None
    technical = next((s for s in report.agent_signals if s.agent == "technical"), None)
    ml_rank_score = None
    sector_rank = None
    if technical:
        ml_rank_score = technical.raw.get("ml_rank_score")
        if ml_rank_score is None:
            ml_rank_score = technical.raw.get("cs_prob")
        sector_rank = technical.raw.get("sector_rank")
    if get_predict_backend() == "ml":
        model_name = get_ml_model_name()
        try:
            from tradingai.analysis.ml.artifacts import get_manifest_version

            model_version = get_manifest_version()
        except Exception:
            model_version = report.fusion.get("model_version")
    rows = []
    for horizon in report.horizons:
        rows.append(
            {
                "ticker": report.ticker.upper(),
                "timestamp": timestamp,
                "horizon_days": horizon.horizon_days,
                "probability_up": horizon.probability_up,
                "expected_return_pct": horizon.expected_return_pct,
                "confidence": report.confidence,
                "combined_score": report.fusion["combined_score"],
                "sector": metadata.sector,
                "regime": metadata.regime,
                "technical_score": report.fusion["technical_score"],
                "news_score": report.fusion["news_score"],
                "regime_score": report.fusion["regime_score"],
                "catalyst_tags": metadata.catalyst_tags,
                "reason_tags": metadata.reason_tags,
                "prediction_type": metadata.prediction_type,
                "agent_payload": agent_payload,
                "reasons": report.reasons,
                "model_name": model_name,
                "model_version": model_version,
                "ml_rank_score": ml_rank_score,
                "sector_rank": sector_rank,
            }
        )
    return predictions_repo.save_prediction_rows(rows)


def get_prediction_history(ticker: str | None = None, *, limit: int = 20) -> list[dict]:
    return predictions_repo.get_prediction_history(ticker, limit=limit)


def get_predictions_pending_outcomes(horizon_days: int) -> list[dict]:
    return predictions_repo.get_pending_predictions(horizon_days)


def save_outcome(
    prediction_id: int,
    actual_return_pct: float,
    *,
    probability_up: float,
) -> None:
    predictions_repo.update_prediction_outcome(
        prediction_id,
        actual_return_pct,
        probability_up,
    )


def get_evaluated_rows(horizon_days: int | None = None) -> list[dict]:
    return predictions_repo.get_evaluated_predictions(horizon_days)


def get_track_record(**filters) -> dict:
    return predictions_repo.get_track_record(**filters)


def get_segment_calibration(horizon_days: int | None = None) -> list[dict]:
    return predictions_repo.get_segment_calibration(horizon_days=horizon_days)
