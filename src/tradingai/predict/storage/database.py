from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tradingai.predict.config import get_database_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    as_of TEXT NOT NULL,
    technical_score REAL NOT NULL,
    news_score REAL NOT NULL,
    regime_score REAL NOT NULL,
    combined_score REAL NOT NULL,
    confidence REAL NOT NULL,
    agent_payload_json TEXT NOT NULL,
    horizons_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    horizon_days INTEGER NOT NULL,
    actual_return_pct REAL NOT NULL,
    direction_correct INTEGER NOT NULL,
    evaluated_at TEXT NOT NULL,
    UNIQUE(prediction_id, horizon_days),
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);
"""


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_database_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str | None = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
