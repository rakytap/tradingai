from __future__ import annotations

import json
from datetime import date, datetime, timezone

from tradingai.data.connection import get_connection


def save_prediction_rows(
    rows: list[dict],
    *,
    database_url: str | None = None,
) -> list[int]:
    ids: list[int] = []
    with get_connection(database_url) as conn:
        for row in rows:
            result = conn.execute(
                """
                INSERT INTO predictions (
                    ticker, timestamp, horizon_days,
                    probability_up, expected_return_pct, confidence, combined_score,
                    sector, regime, technical_score, news_score, regime_score,
                    catalyst_tags, reason_tags, prediction_type,
                    agent_payload, reasons,
                    model_name, model_version
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s::jsonb, %s::jsonb,
                    %s, %s
                )
                ON CONFLICT (ticker, timestamp, horizon_days) DO UPDATE SET
                    probability_up = EXCLUDED.probability_up,
                    expected_return_pct = EXCLUDED.expected_return_pct,
                    confidence = EXCLUDED.confidence,
                    combined_score = EXCLUDED.combined_score,
                    sector = EXCLUDED.sector,
                    regime = EXCLUDED.regime,
                    technical_score = EXCLUDED.technical_score,
                    news_score = EXCLUDED.news_score,
                    regime_score = EXCLUDED.regime_score,
                    catalyst_tags = EXCLUDED.catalyst_tags,
                    reason_tags = EXCLUDED.reason_tags,
                    prediction_type = EXCLUDED.prediction_type,
                    agent_payload = EXCLUDED.agent_payload,
                    reasons = EXCLUDED.reasons,
                    model_name = EXCLUDED.model_name,
                    model_version = EXCLUDED.model_version
                RETURNING id
                """,
                (
                    row["ticker"],
                    row["timestamp"],
                    row["horizon_days"],
                    row["probability_up"],
                    row["expected_return_pct"],
                    row["confidence"],
                    row["combined_score"],
                    row.get("sector"),
                    row.get("regime"),
                    row.get("technical_score"),
                    row.get("news_score"),
                    row.get("regime_score"),
                    row.get("catalyst_tags", []),
                    row.get("reason_tags", []),
                    row.get("prediction_type"),
                    json.dumps(row.get("agent_payload", {})),
                    json.dumps(row.get("reasons", [])),
                    row.get("model_name"),
                    row.get("model_version"),
                ),
            ).fetchone()
            ids.append(int(result[0]))
    return ids


def get_prediction_history(
    ticker: str | None = None,
    *,
    limit: int = 20,
    database_url: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM predictions"
    params: list[object] = []
    if ticker:
        query += " WHERE ticker = %s"
        params.append(ticker.upper())
    query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)

    with get_connection(database_url) as conn:
        cur = conn.execute(query, params)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def get_pending_predictions(
    horizon_days: int,
    *,
    database_url: str | None = None,
) -> list[dict]:
    with get_connection(database_url) as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE horizon_days = %s
              AND actual_return_pct IS NULL
            ORDER BY timestamp ASC
            """,
            (horizon_days,),
        )
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def update_prediction_outcome(
    prediction_id: int,
    actual_return_pct: float,
    probability_up: float,
    *,
    database_url: str | None = None,
) -> None:
    from tradingai.analysis.ml.direction import direction_is_correct

    direction_correct = direction_is_correct(probability_up, actual_return_pct)
    evaluated_at = datetime.now(timezone.utc)
    with get_connection(database_url) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET actual_return_pct = %s,
                direction_correct = %s,
                evaluated_at = %s
            WHERE id = %s
            """,
            (actual_return_pct, direction_correct, evaluated_at, prediction_id),
        )


def get_evaluated_predictions(
    horizon_days: int | None = None,
    *,
    database_url: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM predictions WHERE actual_return_pct IS NOT NULL"
    params: list[object] = []
    if horizon_days is not None:
        query += " AND horizon_days = %s"
        params.append(horizon_days)
    query += " ORDER BY timestamp DESC"

    with get_connection(database_url) as conn:
        cur = conn.execute(query, params)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def get_track_record(
    *,
    sector: str | None = None,
    prediction_type: str | None = None,
    catalyst_tag: str | None = None,
    regime: str | None = None,
    horizon_days: int | None = None,
    database_url: str | None = None,
) -> dict:
    query = """
        SELECT
            COUNT(*) AS sample_size,
            AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) AS hit_rate,
            AVG(actual_return_pct) AS avg_return,
            AVG(probability_up) AS avg_predicted_prob
        FROM predictions
        WHERE actual_return_pct IS NOT NULL
    """
    params: list[object] = []
    if sector:
        query += " AND sector ILIKE %s"
        params.append(f"%{sector}%")
    if prediction_type:
        query += " AND prediction_type = %s"
        params.append(prediction_type)
    if catalyst_tag:
        query += " AND %s = ANY(catalyst_tags)"
        params.append(catalyst_tag)
    if regime:
        query += " AND regime = %s"
        params.append(regime)
    if horizon_days is not None:
        query += " AND horizon_days = %s"
        params.append(horizon_days)

    with get_connection(database_url) as conn:
        row = conn.execute(query, params).fetchone()

    sample_size = int(row[0] or 0)
    hit_rate = float(row[1]) if row[1] is not None else None
    avg_return = float(row[2]) if row[2] is not None else None
    avg_predicted_prob = float(row[3]) if row[3] is not None else None
    overconfidence = None
    if hit_rate is not None and avg_predicted_prob is not None:
        overconfidence = round(avg_predicted_prob - hit_rate, 4)

    return {
        "sample_size": sample_size,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "avg_return_pct": round(avg_return, 4) if avg_return is not None else None,
        "avg_predicted_prob": round(avg_predicted_prob, 4) if avg_predicted_prob is not None else None,
        "overconfidence": overconfidence,
    }


def get_segment_calibration(
    *,
    horizon_days: int | None = None,
    database_url: str | None = None,
) -> list[dict]:
    query = """
        SELECT
            sector,
            prediction_type,
            horizon_days,
            COUNT(*) AS sample_size,
            AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) AS hit_rate,
            AVG(actual_return_pct) AS avg_return,
            AVG(probability_up) AS avg_predicted_prob,
            AVG(POWER(probability_up - CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END, 2)) AS brier_score
        FROM predictions
        WHERE actual_return_pct IS NOT NULL
    """
    params: list[object] = []
    if horizon_days is not None:
        query += " AND horizon_days = %s"
        params.append(horizon_days)
    query += """
        GROUP BY sector, prediction_type, horizon_days
        HAVING COUNT(*) >= 1
        ORDER BY sample_size DESC
    """

    with get_connection(database_url) as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        hit_rate = float(row[4]) if row[4] is not None else None
        avg_predicted_prob = float(row[6]) if row[6] is not None else None
        results.append(
            {
                "sector": row[0],
                "prediction_type": row[1],
                "horizon_days": row[2],
                "sample_size": int(row[3]),
                "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
                "avg_return_pct": round(float(row[5]), 4) if row[5] is not None else None,
                "avg_predicted_prob": (
                    round(avg_predicted_prob, 4) if avg_predicted_prob is not None else None
                ),
                "brier_score": round(float(row[7]), 4) if row[7] is not None else None,
                "overconfidence": (
                    round(avg_predicted_prob - hit_rate, 4)
                    if hit_rate is not None and avg_predicted_prob is not None
                    else None
                ),
            }
        )
    return results
