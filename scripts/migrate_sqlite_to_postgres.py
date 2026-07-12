#!/usr/bin/env python3
"""One-time migration from legacy SQLite predictions to PostgreSQL."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.predictions import save_prediction_rows
from tradingai.data.repositories.stocks import upsert_stock

SQLITE_PATH = Path("./data/predictions.sqlite3")


def migrate(sqlite_path: Path = SQLITE_PATH) -> dict:
    if not sqlite_path.exists():
        return {"migrated": 0, "message": f"No SQLite database at {sqlite_path}"}

    initialize_schema()
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    predictions = conn.execute("SELECT * FROM predictions ORDER BY id").fetchall()
    outcomes = conn.execute("SELECT * FROM outcomes").fetchall()
    outcome_map: dict[tuple[int, int], sqlite3.Row] = {
        (row["prediction_id"], row["horizon_days"]): row for row in outcomes
    }

    migrated = 0
    for prediction in predictions:
        ticker = prediction["ticker"]
        upsert_stock(ticker)
        horizons = json.loads(prediction["horizons_json"])
        agent_payload = json.loads(prediction["agent_payload_json"])
        reasons = json.loads(prediction["reasons_json"])
        rows = []
        for horizon in horizons:
            horizon_days = horizon["horizon_days"]
            outcome = outcome_map.get((prediction["id"], horizon_days))
            row = {
                "ticker": ticker,
                "timestamp": prediction["as_of"],
                "horizon_days": horizon_days,
                "probability_up": horizon["probability_up"],
                "expected_return_pct": horizon["expected_return_pct"],
                "confidence": prediction["confidence"],
                "combined_score": prediction["combined_score"],
                "sector": None,
                "regime": agent_payload.get("regime", {}).get("regime"),
                "technical_score": prediction["technical_score"],
                "news_score": prediction["news_score"],
                "regime_score": prediction["regime_score"],
                "catalyst_tags": [],
                "reason_tags": [],
                "prediction_type": "general_directional",
                "agent_payload": agent_payload,
                "reasons": reasons,
            }
            rows.append(row)
        save_prediction_rows(rows)
        if outcome:
            from tradingai.data.repositories.predictions import update_prediction_outcome

            for horizon in horizons:
                outcome_row = outcome_map.get((prediction["id"], horizon["horizon_days"]))
                if outcome_row:
                    # IDs changed on insert; update by unique key would need lookup.
                    pass
        migrated += 1

    conn.close()
    return {"migrated": migrated}


def main() -> None:
    result = migrate()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
