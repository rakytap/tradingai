from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingai.news.analysis.scoring_optimizer import (
    NewsScoringOptimizationError,
    optimize_news_scoring,
    resimulate_news_scores,
)
from tradingai.news.config import (
    DEFAULT_NEWS_SCORING_PATH,
    clear_news_scoring_cache,
    default_news_scoring_params,
    save_news_scoring_params,
)
from tradingai.predict.config import (
    DEFAULT_FUSION_TUNING_PATH,
    DEFAULT_WEIGHTS,
    HORIZON_SCALES,
    clear_fusion_config_cache,
    get_fusion_weights,
    get_horizon_scales,
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


def _print_news_summary(result: dict, *, output: Path, dry_run: bool) -> None:
    print("News scoring optimization complete")
    print(f"  Metric: {result['metric']} (train_fraction={result['train_fraction']})")
    print(f"  Source weights: {result['source_weights']}")
    print(f"  Recency: {result['recency']}")
    baseline = result["baseline_validation"]
    tuned = result["tuned_validation"]
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
    if dry_run:
        print("  Dry run — news scoring file not written.")
    else:
        print(f"  Saved to {output}")


def _print_fusion_summary(result: dict, *, output: Path, dry_run: bool) -> None:
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
        print("  Dry run — fusion weights file not written.")
    else:
        print(f"  Saved to {output}")


def _load_rows(horizon_days: int | None) -> tuple[tuple[int, ...], list[dict]]:
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
    return horizons, rows


def tune_fusion_weights(
    *,
    output: Path,
    news_output: Path,
    train_fraction: float,
    min_samples: int,
    horizon_days: int | None,
    dry_run: bool,
    weight_step: float,
    skip_news: bool = False,
    news_only: bool = False,
) -> dict:
    if skip_news and news_only:
        raise FusionOptimizationError("Cannot combine --skip-news and --news-only.")

    horizons, rows = _load_rows(horizon_days)
    fusion_weights = get_fusion_weights()
    horizon_scales = get_horizon_scales()

    news_result: dict | None = None
    news_params = default_news_scoring_params()

    if not skip_news:
        news_result = optimize_news_scoring(
            rows,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
            train_fraction=train_fraction,
            min_samples=min_samples,
        )
        news_params = {
            "source_weights": news_result["source_weights"],
            "recency": news_result["recency"],
        }
        if not dry_run:
            save_news_scoring_params(news_result, path=news_output)

    if news_only:
        return {
            "news": news_result,
            "fusion": None,
        }

    fusion_rows = resimulate_news_scores(rows, news_params) if not skip_news else rows
    fusion_result = optimize_fusion(
        fusion_rows,
        horizons=horizons,
        train_fraction=train_fraction,
        min_samples=min_samples,
        weight_step=weight_step,
        baseline_weights=DEFAULT_WEIGHTS,
        baseline_scales=dict(HORIZON_SCALES),
    )

    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(fusion_result, indent=2) + "\n", encoding="utf-8")
        clear_fusion_config_cache()
        clear_news_scoring_cache()

    return {
        "news": news_result,
        "fusion": fusion_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune news scoring params (nested), then fusion agent weights and "
            "horizon scales from evaluated predictions."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FUSION_TUNING_PATH,
        help="Path for tuned fusion weights JSON (default: data/fusion_weights.json).",
    )
    parser.add_argument(
        "--news-output",
        type=Path,
        default=DEFAULT_NEWS_SCORING_PATH,
        help="Path for tuned news scoring JSON (default: data/news_scoring.json).",
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
        help="Minimum evaluated rows required (per horizon for fusion; with snapshots for news).",
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
        "--skip-news",
        action="store_true",
        help="Skip news scoring phase (fusion-only, prior behavior).",
    )
    parser.add_argument(
        "--news-only",
        action="store_true",
        help="Tune and write news scoring only; skip fusion phase.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report without writing JSON files.",
    )
    args = parser.parse_args()

    try:
        result = tune_fusion_weights(
            output=args.output,
            news_output=args.news_output,
            train_fraction=args.train_fraction,
            min_samples=args.min_samples,
            horizon_days=args.horizon,
            dry_run=args.dry_run,
            weight_step=args.weight_step,
            skip_news=args.skip_news,
            news_only=args.news_only,
        )
    except (FusionOptimizationError, NewsScoringOptimizationError) as exc:
        raise SystemExit(str(exc)) from exc

    if result.get("news") is not None:
        _print_news_summary(result["news"], output=args.news_output, dry_run=args.dry_run)
    if result.get("fusion") is not None:
        _print_fusion_summary(result["fusion"], output=args.output, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
