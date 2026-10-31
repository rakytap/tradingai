"""Dataset, splits, and batching for dip-cut sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from tradingai.intraday.ml import INDEX_TO_LABEL, LABEL_TO_INDEX
from tradingai.intraday.ml.config import (
    DEFAULT_ALIGN,
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_BARS,
    DEFAULT_VAL_FRACTION,
    N_SEQ_CHANNELS,
    N_STATIC_FEATURES,
    SEQ_CHANNELS,
)

Align = Literal["right", "left"]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for dip-cut training. "
            "Install with: pip install 'tradingai[torch]' or pip install torch"
        ) from exc
    return torch


@dataclass(frozen=True)
class DipSample:
    sample_id: int
    ticker: str
    session_date: str
    label: int
    seq: np.ndarray  # float32 [C, T]
    mask: np.ndarray  # float32 [T]
    static: np.ndarray  # float32 [S]


def load_dip_frames(dataset_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(dataset_dir)
    samples = pd.read_parquet(root / "samples.parquet")
    bars = pd.read_parquet(root / "bars.parquet")
    if samples.empty or bars.empty:
        raise ValueError(f"Empty dip dataset under {root}")
    return samples, bars


def encode_sequence(
    bar_frame: pd.DataFrame,
    prior_close: float,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    align: Align = DEFAULT_ALIGN,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``[C, T]`` features and ``[T]`` mask from cut bars (no prior-close row).

    Price channels are ``price / prior_close - 1``. Volume is ``log1p(volume)``.
    Sequences shorter than ``max_len`` are zero-padded; ``align='right'`` keeps
    bars nearest the cut.
    """
    pc = float(prior_close)
    if not np.isfinite(pc) or pc <= 0:
        raise ValueError("prior_close must be a positive finite price")

    work = bar_frame.loc[~bar_frame["is_prior_close"].astype(bool)].sort_values("seq")
    if work.empty:
        seq = np.zeros((N_SEQ_CHANNELS, max_len), dtype=np.float32)
        mask = np.zeros((max_len,), dtype=np.float32)
        return seq, mask

    opens = work["open"].astype(float).to_numpy()
    highs = work["high"].astype(float).to_numpy()
    lows = work["low"].astype(float).to_numpy()
    closes = work["close"].astype(float).to_numpy()
    volumes = work["volume"].astype(float).to_numpy()

    channels = np.stack(
        [
            opens / pc - 1.0,
            highs / pc - 1.0,
            lows / pc - 1.0,
            closes / pc - 1.0,
            np.log1p(np.maximum(volumes, 0.0)),
        ],
        axis=0,
    ).astype(np.float32)
    channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)

    t = channels.shape[1]
    if t > max_len:
        if align == "right":
            channels = channels[:, -max_len:]
        else:
            channels = channels[:, :max_len]
        t = max_len

    seq = np.zeros((N_SEQ_CHANNELS, max_len), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    if align == "right":
        seq[:, max_len - t :] = channels
        mask[max_len - t :] = 1.0
    else:
        seq[:, :t] = channels
        mask[:t] = 1.0
    return seq, mask


def encode_static(
    *,
    prior_close: float,
    sell_crossings_5d: float,
    cut_end,
) -> np.ndarray:
    """Static features: log(prior_close), log1p(crossings), minutes/390.

    Cut side (before/after the dip) is not a model input; it only drives
    training labels when building the dataset.
    """
    stamp = pd.Timestamp(cut_end)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    minutes = stamp.hour * 60 + stamp.minute - (9 * 60 + 30)
    minutes = float(np.clip(minutes, 0.0, 390.0))
    return np.asarray(
        [
            np.log(max(float(prior_close), 1e-6)),
            np.log1p(max(float(sell_crossings_5d), 0.0)),
            minutes / 390.0,
        ],
        dtype=np.float32,
    )


def build_samples(
    samples: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    min_bars: int = DEFAULT_MIN_BARS,
    align: Align = DEFAULT_ALIGN,
) -> list[DipSample]:
    """Materialize encoded samples; drops cuts shorter than ``min_bars``."""
    grouped = {
        int(sid): frame
        for sid, frame in bars.groupby("sample_id", sort=False)
    }
    out: list[DipSample] = []
    for row in samples.itertuples(index=False):
        sid = int(row.sample_id)
        frame = grouped.get(sid)
        if frame is None:
            continue
        n_real = int((~frame["is_prior_close"].astype(bool)).sum())
        if n_real < int(min_bars):
            continue
        label = int(row.label)
        if label not in LABEL_TO_INDEX:
            continue
        seq, mask = encode_sequence(
            frame,
            float(row.prior_close),
            max_len=max_len,
            align=align,
        )
        static = encode_static(
            prior_close=float(row.prior_close),
            sell_crossings_5d=float(row.sell_crossings_5d),
            cut_end=row.cut_end,
        )
        assert static.shape[0] == N_STATIC_FEATURES
        out.append(
            DipSample(
                sample_id=sid,
                ticker=str(row.ticker),
                session_date=str(row.session_date),
                label=label,
                seq=seq,
                mask=mask,
                static=static,
            )
        )
    return out


def chronological_split(
    samples: list[DipSample],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
) -> tuple[list[DipSample], list[DipSample]]:
    """Hold out the latest session dates (fraction of unique dates).

    Within the boundary date, samples are shuffled with ``seed`` into train/val
    only if a single date remains; otherwise the split is strictly by date.
    """
    if not samples:
        return [], []
    frac = float(val_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    dates = sorted({s.session_date for s in samples})
    if len(dates) == 1:
        rng = np.random.default_rng(seed)
        order = np.arange(len(samples))
        rng.shuffle(order)
        n_val = max(1, int(round(len(samples) * frac)))
        val_idx = set(order[:n_val].tolist())
        train = [samples[i] for i in range(len(samples)) if i not in val_idx]
        val = [samples[i] for i in range(len(samples)) if i in val_idx]
        return train, val

    n_val_dates = max(1, int(round(len(dates) * frac)))
    val_dates = set(dates[-n_val_dates:])
    train = [s for s in samples if s.session_date not in val_dates]
    val = [s for s in samples if s.session_date in val_dates]
    if not train or not val:
        # Degenerate: fall back to random split.
        rng = np.random.default_rng(seed)
        order = np.arange(len(samples))
        rng.shuffle(order)
        n_val = max(1, int(round(len(samples) * frac)))
        val_idx = set(order[:n_val].tolist())
        train = [samples[i] for i in range(len(samples)) if i not in val_idx]
        val = [samples[i] for i in range(len(samples)) if i in val_idx]
    return train, val


def class_weights(samples: list[DipSample]) -> np.ndarray:
    """Inverse-frequency weights ordered as indices 0,1,2 for labels -1,0,1."""
    counts = np.zeros(3, dtype=np.float64)
    for sample in samples:
        counts[LABEL_TO_INDEX[sample.label]] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights.astype(np.float32)


class DipCutDataset:
    """PyTorch dataset wrapping encoded :class:`DipSample` rows."""

    def __init__(self, samples: list[DipSample]):
        self.samples = list(samples)
        _require_torch()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch = _require_torch()
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "seq": torch.from_numpy(sample.seq),
            "mask": torch.from_numpy(sample.mask),
            "static": torch.from_numpy(sample.static),
            "y": torch.tensor(LABEL_TO_INDEX[sample.label], dtype=torch.long),
            "label": sample.label,
        }


def collate_dip_batch(batch: list[dict]):
    torch = _require_torch()
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "seq": torch.stack([item["seq"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "static": torch.stack([item["static"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "label": [item["label"] for item in batch],
    }


def label_names() -> list[str]:
    return [str(INDEX_TO_LABEL[i]) for i in range(3)]
