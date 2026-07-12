from __future__ import annotations

import json
from datetime import datetime, timezone

from tradingai.predict.models import PredictionReport
from tradingai.predict.storage.database import _connect, initialize_database


def save_prediction(report: PredictionReport, db_path: str | None = None) -> int:
    initialize_database(db_path)
    as_of = report.as_of.astimezone(timezone.utc)
    prediction_date = as_of.date().isoformat()
    agent_payload = {
        signal.agent: signal.raw for signal in report.agent_signals
    }

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                ticker, prediction_date, as_of,
                technical_score, news_score, regime_score,
                combined_score, confidence,
                agent_payload_json, horizons_json, reasons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.ticker,
                prediction_date,
                as_of.isoformat(),
                report.fusion["technical_score"],
                report.fusion["news_score"],
                report.fusion["regime_score"],
                report.fusion["combined_score"],
                report.confidence,
                json.dumps(agent_payload),
                json.dumps([horizon.model_dump() for horizon in report.horizons]),
                json.dumps(report.reasons),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_prediction_history(
    ticker: str | None = None,
    *,
    limit: int = 20,
    db_path: str | None = None,
) -> list[dict]:
    initialize_database(db_path)
    query = "SELECT * FROM predictions"
    params: list[object] = []
    if ticker:
        query += " WHERE ticker = ?"
        params.append(ticker.upper())
    query += " ORDER BY as_of DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_predictions_pending_outcomes(
    horizon_days: int,
    *,
    db_path: str | None = None,
) -> list[dict]:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.*
            FROM predictions p
            LEFT JOIN outcomes o
                ON o.prediction_id = p.id AND o.horizon_days = ?
            WHERE o.id IS NULL
            ORDER BY p.prediction_date ASC
            """,
            (horizon_days,),
        ).fetchall()
        return [dict(row) for row in rows]


def save_outcome(
    prediction_id: int,
    horizon_days: int,
    actual_return_pct: float,
    *,
    db_path: str | None = None,
) -> None:
    initialize_database(db_path)
    direction_correct = 1 if actual_return_pct > 0 else 0
    evaluated_at = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO outcomes (
                prediction_id, horizon_days, actual_return_pct,
                direction_correct, evaluated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id, horizon_days) DO UPDATE SET
                actual_return_pct = excluded.actual_return_pct,
                direction_correct = excluded.direction_correct,
                evaluated_at = excluded.evaluated_at
            """,
            (
                prediction_id,
                horizon_days,
                actual_return_pct,
                direction_correct,
                evaluated_at,
            ),
        )
        conn.commit()


def get_evaluated_rows(
    horizon_days: int | None = None,
    *,
    db_path: str | None = None,
) -> list[dict]:
    initialize_database(db_path)
    query = """
        SELECT
            p.id AS prediction_id,
            p.ticker,
            p.prediction_date,
            p.combined_score,
            p.confidence,
            p.horizons_json,
            o.horizon_days,
            o.actual_return_pct,
            o.direction_correct,
            o.evaluated_at
        FROM predictions p
        JOIN outcomes o ON o.prediction_id = p.id
    """
    params: list[object] = []
    if horizon_days is not None:
        query += " WHERE o.horizon_days = ?"
        params.append(horizon_days)
    query += " ORDER BY p.prediction_date DESC"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
