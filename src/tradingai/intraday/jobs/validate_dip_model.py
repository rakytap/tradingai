"""CLI: validate a trained dip-cut classifier checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingai.intraday.ml.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_NUM_WORKERS,
)
from tradingai.intraday.ml.validate import default_checkpoint_path, validate_dip_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a dip-cut classifier; prints metrics and optional JSON."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint.pt (default: data/dip_models/checkpoint.pt)",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="Which chronological partition to score",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write the full validation report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint or default_checkpoint_path(DEFAULT_MODEL_DIR)
    if not Path(checkpoint).is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    result = validate_dip_classifier(
        checkpoint=checkpoint,
        dataset_dir=args.dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        out_json=args.out_json,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
