"""PyTorch models for open-dip cut classification."""

from __future__ import annotations

__all__ = [
    "DipCutClassifier",
    "DipCutDataset",
    "LABEL_TO_INDEX",
    "INDEX_TO_LABEL",
]

LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_LABEL = {0: -1, 1: 0, 2: 1}


def __getattr__(name: str):
    if name == "DipCutClassifier":
        from tradingai.intraday.ml.model import DipCutClassifier

        return DipCutClassifier
    if name == "DipCutDataset":
        from tradingai.intraday.ml.data import DipCutDataset

        return DipCutDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
