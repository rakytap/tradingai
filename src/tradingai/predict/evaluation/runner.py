from __future__ import annotations

import json

from tradingai.data.jobs.sync_daily_prices import sync_daily_prices
from tradingai.predict.config import get_prediction_horizons
from tradingai.predict.evaluation.metrics import compute_evaluation_metrics
from tradingai.predict.evaluation.outcomes import evaluate_pending_outcomes


def run_evaluation() -> dict:
    price_sync = sync_daily_prices()
    results = [evaluate_pending_outcomes(days) for days in get_prediction_horizons()]
    metrics = compute_evaluation_metrics()
    return {
        "price_sync": price_sync,
        "evaluation": results,
        "metrics": metrics,
    }


def main() -> None:
    payload = run_evaluation()
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
