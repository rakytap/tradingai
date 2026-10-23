"""Validation metrics for 3-class dip labels."""

from __future__ import annotations

import numpy as np

from tradingai.intraday.ml import INDEX_TO_LABEL


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            mat[t, p] += 1
    return mat


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    mat = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    f1 = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = float(mat[c, c])
        fp = float(mat[:, c].sum() - tp)
        fn = float(mat[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1[c] = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return f1


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    accuracy = float((y_true == y_pred).mean()) if n else 0.0
    f1 = per_class_f1(y_true, y_pred)
    mat = confusion_matrix(y_true, y_pred)
    report: dict = {
        "n": int(n),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(float(f1.mean()), 6),
        "per_class_f1": {
            str(INDEX_TO_LABEL[i]): round(float(f1[i]), 6) for i in range(len(f1))
        },
        "confusion_matrix": mat.tolist(),
        "support": {
            str(INDEX_TO_LABEL[i]): int(mat[i, :].sum()) for i in range(mat.shape[0])
        },
    }
    if y_prob is not None and len(y_prob):
        probs = np.asarray(y_prob, dtype=float)
        # Negative log-likelihood / cross-entropy on true class.
        rows = np.arange(len(y_true))
        p_true = np.clip(probs[rows, y_true], 1e-12, 1.0)
        report["nll"] = round(float(-np.log(p_true).mean()), 6)
        report["mean_max_prob"] = round(float(probs.max(axis=1).mean()), 6)
    return report
