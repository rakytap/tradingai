from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tradingai.data.repositories.daily_prices import get_forward_return
from tradingai.predict.storage.repository import (
    get_predictions_pending_outcomes,
    save_outcome,
)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def evaluate_pending_outcomes(
    horizon_days: int,
    *,
    as_of: datetime | None = None,
) -> dict:
    as_of = _to_utc(as_of) if as_of else datetime.now(timezone.utc)
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
        prediction_dt = _to_utc(prediction_dt)
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
