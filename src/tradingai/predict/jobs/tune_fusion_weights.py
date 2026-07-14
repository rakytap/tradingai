from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingai.predict.config import (
    DEFAULT_WEIGHTS,
    HORIZON_SCALES,
    clear_fusion_config_cache,
    get_prediction_horizons,
)
from tradingai.predict.fusion.optimizer import FusionOptimizationError, optimize_fusion
from tradingai.predict.storage.repository import get_evaluated_rows


def _format_weights(weights: dict[str, float]) -> str:
    return (
        f"{weights['technical']:.4f},"
        f"{weights['news']:.4f},"
        f"{weights['regime']:.4f}"
    )


def _print_summary(result: dict, *, output: Path, dry_run: bool) -> None:
    weights = result["weights"]
    scales = result["horizon_scales"]
    baseline = result["baseline_validation"]
    tuned = result["tuned_validation"]

    print("Fusion weight optimization complete")
    print(f"  Metric: {result['metric']} (train_fraction={result['train_fraction']})")
    print(f"  Weights: technical={weights['technical']}, news={weights['news']}, regime={weights['regime']}")
    print(f"  Horizon scales: {scales}")
    print(
        "  Validation baseline: "
        f"brier={baseline.get('brier')} "
        f"direction_accuracy={baseline.get('direction_accuracy')} "
        f"n={baseline.get('sample_size')}"
    )
    print(
        "  Validation tuned:    "
        f"brier={tuned.get('brier')} "
        f"direction_accuracy={tuned.get('direction_accuracy')} "
        f"n={tuned.get('sample_size')}"
    )
    print(f"  Reference env line: TRADINGAI_PREDICT_WEIGHTS={_format_weights(weights)}")
    if dry_run:
        print("  Dry run — no file written.")
    else:
        print(f"  Saved to {output}")


def tune_fusion_weights(
    *,
    output: Path,
    train_fraction: float,
    min_samples: int,
    horizon_days: int | None,
    dry_run: bool,
    weight_step: float,
) -> dict:
    horizons = get_prediction_horizons()
    if horizon_days is not None:
        if horizon_days not in horizons:
            raise FusionOptimizationError(
                f"Horizon {horizon_days}d is not in configured horizons {horizons}."
            )
        horizons = (horizon_days,)

    rows: list[dict] = []
    for horizon in horizons:
        rows.extend(get_evaluated_rows(horizon))

    result = optimize_fusion(
        rows,
        horizons=horizons,
        train_fraction=train_fraction,
        min_samples=min_samples,
        weight_step=weight_step,
        baseline_weights=DEFAULT_WEIGHTS,
        baseline_scales=dict(HORIZON_SCALES),
    )

    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        clear_fusion_config_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune fusion agent weights and horizon scales from evaluated predictions."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fusion_weights.json"),
        help="Path for tuned weights JSON (default: data/fusion_weights.json).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Chronological train fraction for scale/weight search (default 0.7).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum evaluated rows required per horizon (default 30).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Tune only this horizon in trading days (default: all configured horizons).",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.05,
        help="Grid step for agent weight search on the simplex (default 0.05).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report without writing fusion_weights.json.",
    )
    args = parser.parse_args()

    try:
        result = tune_fusion_weights(
            output=args.output,
            train_fraction=args.train_fraction,
            min_samples=args.min_samples,
            horizon_days=args.horizon,
            dry_run=args.dry_run,
            weight_step=args.weight_step,
        )
    except FusionOptimizationError as exc:
        raise SystemExit(str(exc)) from exc

    _print_summary(result, output=args.output, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
