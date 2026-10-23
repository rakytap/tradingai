"""Training loop for :class:`DipCutClassifier`."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tradingai.intraday.ml.config import (
    DEFAULT_ALIGN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLOCKS,
    DEFAULT_DATASET_DIR,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_KERNEL,
    DEFAULT_LR,
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_BARS,
    DEFAULT_MODEL_DIR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
)
from tradingai.intraday.ml.data import (
    Align,
    DipCutDataset,
    build_samples,
    chronological_split,
    class_weights,
    collate_dip_batch,
    load_dip_frames,
)
from tradingai.intraday.ml.model import DipCutClassifier
from tradingai.intraday.ml.validate import evaluate_loader

# Hyperparameters that must match the checkpoint when resuming (data + architecture).
_RESUME_LOCK_KEYS = (
    "max_len",
    "min_bars",
    "align",
    "val_fraction",
    "seed",
    "hidden",
    "blocks",
    "kernel",
    "dropout",
)


@dataclass
class TrainConfig:
    dataset_dir: Path = DEFAULT_DATASET_DIR
    out_dir: Path = DEFAULT_MODEL_DIR
    max_len: int = DEFAULT_MAX_LEN
    min_bars: int = DEFAULT_MIN_BARS
    align: Align = DEFAULT_ALIGN
    val_fraction: float = DEFAULT_VAL_FRACTION
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    hidden: int = DEFAULT_HIDDEN
    blocks: int = DEFAULT_BLOCKS
    kernel: int = DEFAULT_KERNEL
    dropout: float = DEFAULT_DROPOUT
    patience: int = DEFAULT_PATIENCE
    num_workers: int = DEFAULT_NUM_WORKERS
    seed: int = DEFAULT_SEED
    device: str = "cpu"
    resume: Path | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _config_public_dict(config: TrainConfig) -> dict[str, Any]:
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(config).items()
        if k != "resume"
    }


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    import torch

    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        static = batch["static"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(seq, mask, static)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * int(y.size(0))
        n += int(y.size(0))
    return total_loss / max(n, 1)


def save_checkpoint(
    out_dir: Path,
    *,
    model: DipCutClassifier,
    config: TrainConfig,
    metrics: dict[str, Any] | None,
    history: list[dict[str, Any]],
    class_weight_vec: list[float],
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_f1: float = -1.0,
    best_state: dict | None = None,
    stale: int = 0,
) -> Path:
    """Write a resumable checkpoint (model, optim, schedule, progress)."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = best_state if best_state is not None else {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    ckpt = {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "best_model_state": best,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_f1": float(best_f1),
        "stale": int(stale),
        "config": {
            **_config_public_dict(config),
            "class_weights": class_weight_vec,
        },
        "metrics": metrics,
        "history": history,
    }
    path = out_dir / "checkpoint.pt"
    torch.save(ckpt, path)
    if metrics is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps(ckpt["config"], indent=2) + "\n")
    return path


def load_training_checkpoint(path: Path | str) -> dict[str, Any]:
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise ValueError(f"Not a dip-model checkpoint: {path}")
    return payload


def _apply_resume_locked_hparams(cfg: TrainConfig, ckpt_config: dict[str, Any]) -> None:
    """Overwrite architecture / split fields from the checkpoint for a consistent resume."""
    for key in _RESUME_LOCK_KEYS:
        if key not in ckpt_config:
            continue
        value = ckpt_config[key]
        if key in {"max_len", "min_bars", "seed", "hidden", "blocks", "kernel"}:
            setattr(cfg, key, int(value))
        elif key in {"val_fraction", "dropout"}:
            setattr(cfg, key, float(value))
        elif key == "align":
            setattr(cfg, key, value)


def train_dip_classifier(config: TrainConfig | None = None) -> dict[str, Any]:
    """Load dataset, train for ``epochs``, keep best-val weights, write artifacts.

    Pass ``config.resume`` (or CLI ``--resume``) to continue from an exported
    ``checkpoint.pt``. ``epochs`` is the **total** epoch budget (not additional).
    ``patience`` only affects the LR scheduler (ReduceLROnPlateau), not stopping.
    """
    import torch
    from torch.utils.data import DataLoader

    cfg = config or TrainConfig()
    resume_payload: dict[str, Any] | None = None
    if cfg.resume is not None:
        resume_path = Path(cfg.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_payload = load_training_checkpoint(resume_path)
        ckpt_cfg = resume_payload.get("config") or {}
        _apply_resume_locked_hparams(cfg, ckpt_cfg)
        print(f"resuming from {resume_path} (epoch {resume_payload.get('epoch', 0)})", flush=True)

    set_seed(cfg.seed)
    device = _device(cfg.device)

    samples_df, bars_df = load_dip_frames(cfg.dataset_dir)
    encoded = build_samples(
        samples_df,
        bars_df,
        max_len=cfg.max_len,
        min_bars=cfg.min_bars,
        align=cfg.align,
    )
    if len(encoded) < 10:
        raise ValueError(f"Too few usable samples after filters: {len(encoded)}")

    train_samples, val_samples = chronological_split(
        encoded,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    if not train_samples or not val_samples:
        raise ValueError("Train/val split produced an empty partition")

    if resume_payload and resume_payload.get("config", {}).get("class_weights"):
        weights = np.asarray(resume_payload["config"]["class_weights"], dtype=np.float32)
    else:
        weights = class_weights(train_samples)

    train_loader = DataLoader(
        DipCutDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_dip_batch,
    )
    val_loader = DataLoader(
        DipCutDataset(val_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_dip_batch,
    )

    model = DipCutClassifier(
        hidden=cfg.hidden,
        blocks=cfg.blocks,
        kernel=cfg.kernel,
        dropout=cfg.dropout,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, cfg.patience // 3)
    )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_state: dict | None = None
    best_val: dict[str, Any] = {}
    start_epoch = 0

    if resume_payload is not None:
        # Continue from last weights when available; fall back to best / model_state.
        last_state = resume_payload.get("last_model_state") or resume_payload.get("model_state")
        model.load_state_dict(last_state)
        if resume_payload.get("best_model_state") is not None:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in resume_payload["best_model_state"].items()
            }
        elif resume_payload.get("model_state") is not None:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in resume_payload["model_state"].items()
            }
        if resume_payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            # Allow CLI --lr to override resumed param-group LRs.
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr
        if resume_payload.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(resume_payload["scheduler_state"])
            except Exception:
                print("warning: could not restore scheduler state; continuing with fresh scheduler", flush=True)
        start_epoch = int(resume_payload.get("epoch") or 0)
        best_f1 = float(resume_payload.get("best_f1", -1.0))
        history = list(resume_payload.get("history") or [])
        prior_metrics = resume_payload.get("metrics") or {}
        if isinstance(prior_metrics.get("val"), dict):
            best_val = dict(prior_metrics["val"])

    if int(cfg.epochs) <= start_epoch:
        raise ValueError(
            f"--epochs ({cfg.epochs}) must be greater than resumed epoch ({start_epoch}); "
            "increase --epochs to continue training"
        )

    for epoch in range(start_epoch + 1, int(cfg.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_loader(model, val_loader, device, criterion=criterion)
        scheduler.step(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics.get('loss')}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1 = float(val_metrics["macro_f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = dict(val_metrics)

        save_checkpoint(
            cfg.out_dir,
            model=model,
            config=cfg,
            metrics=None,
            history=history,
            class_weight_vec=weights.tolist(),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_f1=best_f1,
            best_state=best_state,
            stale=0,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate_loader(model, train_loader, device)
    final = {
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_total_encoded": len(encoded),
        "class_weights": weights.tolist(),
        "train": train_metrics,
        "val": best_val or evaluate_loader(model, val_loader, device),
        "best_val_macro_f1": best_f1,
        "epochs_ran": len(history),
        "resumed_from_epoch": start_epoch,
    }
    # Final export: model_state = best (for inference); keep last via re-read? 
    # Store best as both model_state and best_model_state; also keep last in last_model_state
    # by reading current file's model_state before overwrite if needed.
    last_payload = None
    ckpt_path = Path(cfg.out_dir) / "checkpoint.pt"
    if ckpt_path.is_file():
        try:
            last_payload = load_training_checkpoint(ckpt_path)
        except Exception:
            last_payload = None
    last_state = (last_payload or {}).get("model_state")

    import torch

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "best_model_state": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "last_model_state": last_state
        or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(history[-1]["epoch"]) if history else start_epoch,
        "best_f1": float(best_f1),
        "stale": 0,
        "config": {
            **_config_public_dict(cfg),
            "class_weights": weights.tolist(),
        },
        "metrics": final,
        "history": history,
    }
    torch.save(final_ckpt, ckpt_path)
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps(final_ckpt["config"], indent=2) + "\n")
    final["checkpoint"] = str(ckpt_path)
    return final
