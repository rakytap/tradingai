from __future__ import annotations

from datetime import datetime, timedelta

from tradingai.data.repositories.daily_prices import get_forward_return
from tradingai.predict.storage.repository import (
    get_predictions_pending_outcomes,
    save_outcome,
)


def evaluate_pending_outcomes(
    horizon_days: int,
    *,
    as_of: datetime | None = None,
) -> dict:
    as_of = as_of or datetime.utcnow()
    pending = get_predictions_pending_outcomes(horizon_days)
    evaluated = 0
    skipped = 0
    errors: list[str] = []

    for row in pending:
        timestamp = row["timestamp"]
        if isinstance(timestamp, str):
            prediction_dt = datetime.fromisoformat(timestamp)
        else:
            prediction_dt = timestamp
        if prediction_dt + timedelta(days=horizon_days + 5) > as_of:
            skipped += 1
            continue
        try:
            actual_return = get_forward_return(
                row["ticker"],
                prediction_dt.date(),
                horizon_days,
            )
            if actual_return is None:
                skipped += 1
                continue
            save_outcome(row["id"], actual_return)
            evaluated += 1
        except Exception as exc:
            errors.append(f"{row['ticker']}:{prediction_dt.date()}:{exc}")
            skipped += 1

    return {
        "horizon_days": horizon_days,
        "evaluated": evaluated,
        "skipped": skipped,
        "errors": errors[:10],
    }
