from __future__ import annotations

import json

from tradingai.predict.config import get_prediction_horizons
from tradingai.predict.evaluation.metrics import compute_evaluation_metrics
from tradingai.predict.evaluation.outcomes import evaluate_pending_outcomes


def run_evaluation(db_path: str | None = None) -> dict:
    results = []
    for horizon_days in get_prediction_horizons():
        results.append(evaluate_pending_outcomes(horizon_days, db_path=db_path))
    metrics = compute_evaluation_metrics(db_path=db_path)
    return {"evaluation": results, "metrics": metrics}


def main() -> None:
    payload = run_evaluation()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
