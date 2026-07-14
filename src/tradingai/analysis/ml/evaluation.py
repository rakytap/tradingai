from __future__ import annotations

import statistics

import numpy as np


def _precision_recall(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_pred = y_prob >= 0.5
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {"precision": precision, "recall": recall}


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_true = y_true.astype(int)
    y_pred = y_prob >= 0.5
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else None
    brier_values = [(float(prob) - float(actual)) ** 2 for prob, actual in zip(y_prob, y_true)]
    brier = statistics.mean(brier_values) if brier_values else None
    pr = _precision_recall(y_true, y_prob)
    return {
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "precision": round(pr["precision"], 4) if pr["precision"] is not None else None,
        "recall": round(pr["recall"], 4) if pr["recall"] is not None else None,
        "sample_size": len(y_true),
    }


def rank_models(metrics_by_model: dict[str, dict[str, float | None]]) -> list[dict]:
    ranked = []
    for model_name, metrics in metrics_by_model.items():
        ranked.append({"model": model_name, **metrics})
    ranked.sort(
        key=lambda item: (
            item.get("brier_score") is None,
            item.get("brier_score", 1.0),
            -(item.get("accuracy") or 0.0),
        )
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked
