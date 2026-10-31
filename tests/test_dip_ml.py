"""Unit tests for dip-cut PyTorch classifier stack."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from tradingai.intraday.ml.data import (
    build_samples,
    chronological_split,
    class_weights,
    encode_sequence,
    encode_static,
)
from tradingai.intraday.ml.metrics import classification_report
from tradingai.intraday.ml.model import DipCutClassifier
from tradingai.intraday.ml.train import TrainConfig, train_dip_classifier
from tradingai.intraday.ml.validate import load_checkpoint, validate_dip_classifier
from tradingai.intraday.dip_dataset import write_dip_dataset

ET = ZoneInfo("America/New_York")


def _toy_dataset(tmp_path: Path, n_per_label: int = 12) -> Path:
    sample_rows = []
    bar_rows = []
    sid = 0
    for day_offset, label in enumerate((-1, 0, 1)):
        session = datetime(2026, 10, 14 + day_offset, tzinfo=ET).date()
        for i in range(n_per_label):
            prior = 100.0 + i
            n_bars = 20 + (i % 5)
            start = datetime(2026, 10, 14 + day_offset, 9, 30, tzinfo=ET)
            index = pd.date_range(start, periods=n_bars, freq="1min", tz=ET)
            # Mild path below prior close.
            close = prior * (0.96 - 0.001 * np.arange(n_bars))
            if label == 1:
                close = close + np.linspace(0, prior * 0.01, n_bars)
            frame = pd.DataFrame(
                {
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.2,
                    "close": close,
                    "volume": np.full(n_bars, 1000.0),
                },
                index=index,
            )
            cut_end = index[-1]
            sample_rows.append(
                {
                    "sample_id": sid,
                    "ticker": f"T{label}{i}",
                    "session_date": session,
                    "cut_end": cut_end,
                    "side": "after" if i % 2 else "before",
                    "prior_close": prior,
                    "entry_close": float(close[-1]),
                    "anchor_time": index[5],
                    "dip_pct": -4.0,
                    "max_future_high": float(close[-1]),
                    "rise_pct": 0.4 if label == 1 else 0.0,
                    "label": label,
                    "target_sell_price": float(close[-1]) * 1.01,
                    "sell_crossings_5d": i % 7,
                }
            )
            bar_rows.append(
                {
                    "sample_id": sid,
                    "seq": 0,
                    "bar_time": index[0] - pd.Timedelta(minutes=1),
                    "open": prior,
                    "high": prior,
                    "low": prior,
                    "close": prior,
                    "volume": 0.0,
                    "is_prior_close": True,
                }
            )
            for seq, (ts, row) in enumerate(frame.iterrows(), start=1):
                bar_rows.append(
                    {
                        "sample_id": sid,
                        "seq": seq,
                        "bar_time": ts,
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "volume": float(row.volume),
                        "is_prior_close": False,
                    }
                )
            sid += 1
    samples = pd.DataFrame(sample_rows)
    bars = pd.DataFrame(bar_rows)
    return write_dip_dataset(samples, bars, tmp_path / "dip_dataset")


def test_encode_sequence_right_align_and_static():
    start = datetime(2026, 10, 14, 9, 30, tzinfo=ET)
    index = pd.date_range(start, periods=5, freq="1min", tz=ET)
    bars = pd.DataFrame(
        {
            "seq": [0, 1, 2, 3, 4, 5],
            "open": [100.0, 99, 98, 97, 96, 95],
            "high": [100.0, 99, 98, 97, 96, 95],
            "low": [100.0, 99, 98, 97, 96, 95],
            "close": [100.0, 99, 98, 97, 96, 95],
            "volume": [0, 10, 10, 10, 10, 10],
            "is_prior_close": [True, False, False, False, False, False],
            "bar_time": [index[0] - pd.Timedelta(minutes=1), *index],
        }
    )
    seq, mask = encode_sequence(bars, prior_close=100.0, max_len=8, align="right")
    assert seq.shape == (5, 8)
    assert mask.sum() == 5
    assert mask[-1] == 1.0
    assert mask[0] == 0.0
    static = encode_static(
        prior_close=100.0,
        sell_crossings_5d=3,
        cut_end=index[-1],
    )
    assert static.shape == (3,)
    assert abs(float(static[0]) - np.log(100.0)) < 1e-5


def test_model_predict_proba_shape():
    import torch

    model = DipCutClassifier(hidden=16, blocks=2, kernel=3, dropout=0.0)
    seq = torch.randn(4, 5, 32)
    mask = torch.ones(4, 32)
    static = torch.randn(4, 3)
    probs = model.predict_proba(seq, mask, static)
    assert probs.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_chronological_split_and_weights():
    from tradingai.intraday.ml.data import DipSample

    samples: list[DipSample] = []
    sid = 0
    for d, lab in (("2026-10-14", -1), ("2026-10-15", 0), ("2026-10-16", 1)):
        for _ in range(5):
            samples.append(
                DipSample(
                    sample_id=sid,
                    ticker="X",
                    session_date=d,
                    label=lab,
                    seq=np.zeros((5, 8), np.float32),
                    mask=np.ones(8, np.float32),
                    static=np.zeros(3, np.float32),
                )
            )
            sid += 1
    train, val = chronological_split(samples, val_fraction=0.34, seed=0)
    assert train and val
    assert {s.session_date for s in val}.isdisjoint({s.session_date for s in train})
    w = class_weights(train)
    assert w.shape == (3,)
    assert np.all(w > 0)


def test_train_and_validate_smoke(tmp_path: Path):
    dataset_dir = _toy_dataset(tmp_path)
    out_dir = tmp_path / "model"
    result = train_dip_classifier(
        TrainConfig(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            max_len=32,
            min_bars=5,
            epochs=2,
            batch_size=8,
            hidden=16,
            blocks=1,
            kernel=3,
            patience=2,
            device="cpu",
            seed=0,
        )
    )
    assert (out_dir / "checkpoint.pt").is_file()
    assert result["n_train"] > 0
    assert result["n_val"] > 0
    assert "val" in result

    report = validate_dip_classifier(
        checkpoint=out_dir / "checkpoint.pt",
        dataset_dir=dataset_dir,
        split="val",
        device="cpu",
        out_json=out_dir / "val_report.json",
    )
    assert report["metrics"]["n"] == result["n_val"]
    assert (out_dir / "val_report.json").is_file()
    model, payload = load_checkpoint(out_dir / "checkpoint.pt")
    assert isinstance(model, DipCutClassifier)
    assert "optimizer_state" in payload
    assert payload.get("epoch", 0) >= 1


def test_resume_training_continues_epochs(tmp_path: Path):
    import torch

    dataset_dir = _toy_dataset(tmp_path)
    out_dir = tmp_path / "model"
    first = train_dip_classifier(
        TrainConfig(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            max_len=32,
            min_bars=5,
            epochs=2,
            batch_size=8,
            hidden=16,
            blocks=1,
            kernel=3,
            patience=5,
            device="cpu",
            seed=0,
        )
    )
    ckpt = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2
    assert first["resumed_from_epoch"] == 0

    second = train_dip_classifier(
        TrainConfig(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            max_len=32,
            min_bars=5,
            epochs=4,
            batch_size=8,
            hidden=16,
            blocks=1,
            kernel=3,
            patience=5,
            device="cpu",
            seed=0,
            resume=out_dir / "checkpoint.pt",
        )
    )
    assert second["resumed_from_epoch"] == 2
    assert second["epochs_ran"] >= 3
    ckpt2 = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] >= 3
    assert len(ckpt2["history"]) == second["epochs_ran"]


def test_classification_report_basic():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_prob = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.5, 0.3],
            [0.6, 0.2, 0.2],
        ]
    )
    report = classification_report(y_true, y_pred, y_prob)
    assert report["n"] == 4
    assert 0.0 <= report["accuracy"] <= 1.0
    assert "macro_f1" in report
