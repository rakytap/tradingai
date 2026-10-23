"""CLI: train the dip-cut 1D ResNet/TCN classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from tradingai.intraday.ml.train import TrainConfig, train_dip_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 1D ResNet/TCN to predict dip-cut labels (-1, 0, 1)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory with samples.parquet and bars.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to write checkpoint.pt and metrics",
    )
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS)
    parser.add_argument(
        "--align",
        choices=("right", "left"),
        default=DEFAULT_ALIGN,
        help="Pad alignment; right keeps bars nearest the cut",
    )
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="ReduceLROnPlateau patience (does not early-stop training)",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu | cuda | auto",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Continue from an exported checkpoint.pt. "
            "Architecture/split hparams are taken from the checkpoint; "
            "--epochs is the total epoch budget (must exceed the resumed epoch)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        max_len=args.max_len,
        min_bars=args.min_bars,
        align=args.align,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        blocks=args.blocks,
        kernel=args.kernel,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
    result = train_dip_classifier(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
