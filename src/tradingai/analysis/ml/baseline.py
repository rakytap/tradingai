from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier


class BaselinePredictor:
    name = "baseline"

    def __init__(self) -> None:
        self._model = DummyClassifier(strategy="most_frequent")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
