"""Validation / evaluation helpers for dip-cut classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tradingai.intraday.ml import INDEX_TO_LABEL
from tradingai.intraday.ml.config import (
    DEFAULT_ALIGN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DIR,
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_BARS,
    DEFAULT_MODEL_DIR,
    DEFAULT_NUM_WORKERS,
)
from tradingai.intraday.ml.data import (
    Align,
    DipCutDataset,
    build_samples,
    chronological_split,
    collate_dip_batch,
    load_dip_frames,
)
from tradingai.intraday.ml.metrics import classification_report
from tradingai.intraday.ml.model import DipCutClassifier


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate_loader(model, loader, device, criterion=None) -> dict[str, Any]:
    """Run ``model`` on ``loader``; return metrics including optional CE loss."""
    import torch

    model.eval()
    ys: list[int] = []
    preds: list[int] = []
    probs: list[np.ndarray] = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            static = batch["static"].to(device)
            y = batch["y"].to(device)
            logits = model(seq, mask, static)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * int(y.size(0))
            prob = torch.softmax(logits, dim=-1)
            pred = prob.argmax(dim=-1)
            ys.extend(y.detach().cpu().tolist())
            preds.extend(pred.detach().cpu().tolist())
            probs.append(prob.detach().cpu().numpy())
            n += int(y.size(0))
    y_true = np.asarray(ys, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    y_prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, 3))
    report = classification_report(y_true, y_pred, y_prob)
    if criterion is not None:
        report["loss"] = round(total_loss / max(n, 1), 6)
    return report


def load_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: str = "cpu",
) -> tuple[DipCutClassifier, dict[str, Any]]:
    import torch

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    model = DipCutClassifier(
        hidden=int(cfg.get("hidden", 64)),
        blocks=int(cfg.get("blocks", 3)),
        kernel=int(cfg.get("kernel", 5)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    # Prefer best validation weights when present (resumable checkpoints keep both).
    state = payload.get("best_model_state") or payload.get("model_state")
    model.load_state_dict(state)
    model.to(_device(device))
    model.eval()
    return model, payload


def validate_dip_classifier(
    *,
    checkpoint: Path | str,
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    split: str = "val",
    val_fraction: float = 0.2,
    max_len: int | None = None,
    min_bars: int | None = None,
    align: Align | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = 0,
    device: str = "cpu",
    out_json: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on train/val/all encoded samples."""
    import torch
    from torch.utils.data import DataLoader

    model, payload = load_checkpoint(checkpoint, device=device)
    cfg = payload.get("config") or {}
    max_len = int(max_len if max_len is not None else cfg.get("max_len", DEFAULT_MAX_LEN))
    min_bars = int(min_bars if min_bars is not None else cfg.get("min_bars", DEFAULT_MIN_BARS))
    align_val: Align = align or cfg.get("align", DEFAULT_ALIGN)  # type: ignore[assignment]
    if align_val not in ("right", "left"):
        align_val = DEFAULT_ALIGN

    samples_df, bars_df = load_dip_frames(dataset_dir)
    encoded = build_samples(
        samples_df,
        bars_df,
        max_len=max_len,
        min_bars=min_bars,
        align=align_val,
    )
    train_samples, val_samples = chronological_split(
        encoded,
        val_fraction=float(cfg.get("val_fraction", val_fraction)),
        seed=int(cfg.get("seed", seed)),
    )
    split_name = split.lower().strip()
    if split_name == "train":
        chosen = train_samples
    elif split_name == "all":
        chosen = encoded
    else:
        chosen = val_samples
        split_name = "val"

    loader = DataLoader(
        DipCutDataset(chosen),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_dip_batch,
    )
    metrics = evaluate_loader(model, loader, _device(device))
    result = {
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "split": split_name,
        "n": len(chosen),
        "label_order": [INDEX_TO_LABEL[i] for i in range(3)],
        "metrics": metrics,
    }
    if out_json:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        result["out_json"] = str(path)
    return result


def predict_batch(
    model: DipCutClassifier,
    seq,
    mask,
    static,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Return ``[B, 3]`` probabilities for labels ``(-1, 0, 1)``."""
    import torch

    dev = _device(device)
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(
            seq.to(dev),
            mask.to(dev),
            static.to(dev),
        )
    return probs.detach().cpu().numpy()


def default_checkpoint_path(model_dir: Path | str = DEFAULT_MODEL_DIR) -> Path:
    return Path(model_dir) / "checkpoint.pt"
