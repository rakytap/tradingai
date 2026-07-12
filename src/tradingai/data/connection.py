from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import psycopg

from tradingai.data.config import get_database_url

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@contextmanager
def get_connection(database_url: str | None = None):
    conn = psycopg.connect(database_url or get_database_url())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_schema(database_url: str | None = None) -> None:
    sql = _SCHEMA_PATH.read_text()
    with get_connection(database_url) as conn:
        conn.execute(sql)
