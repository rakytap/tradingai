from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingai.news.analysis.event_graph_optimizer import (
    EventGraphOptimizationError,
    optimize_event_graph,
    resimulate_fused_news_scores,
)
from tradingai.news.analysis.l3_spill_optimizer import (
    L3SpillOptimizationError,
    optimize_l3_spill_alpha,
)
from tradingai.news.analysis.scoring_optimizer import (
    NewsScoringOptimizationError,
    optimize_news_scoring,
)
from tradingai.news.config import (
    DEFAULT_EVENT_GRAPH_PATH,
    DEFAULT_NEWS_SCORING_PATH,
    clear_event_graph_cache,
    clear_news_scoring_cache,
    default_news_scoring_params,
    get_l3_news_spill_alpha,
    save_event_graph_params,
    save_l3_spill_alpha,
    save_news_scoring_params,
)
from tradingai.news.analysis.events import default_event_graph_params
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


def _print_events_summary(result: dict, *, output: Path, dry_run: bool) -> None:
    print("Event-graph optimization complete")
    print(f"  Metric: {result['metric']} (train_fraction={result['train_fraction']})")
    print(
        "  Knobs: "
        f"decay={result['propagation_decay']} "
        f"divisor={result['score_divisor']} "
        f"edge_scale={result['edge_scale']} "
        f"max_hops={result['max_hops']}"
    )
    enabled = sum(1 for spec in result["event_specs"] if spec.get("enabled", True))
    print(f"  Event specs enabled: {enabled}/{len(result['event_specs'])}")
    samples = result.get("sample_sizes") or {}
    print(
        "  Samples: "
        f"eligible={samples.get('eligible')} "
        f"corpus={samples.get('with_event_text_corpus')} "
        f"train={samples.get('train')} "
        f"val={samples.get('validation')}"
    )
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
        print("  Dry run — event graph file not written.")
    else:
        print(f"  Saved to {output}")


def _print_l3_summary(result: dict, *, output: Path, dry_run: bool) -> None:
    print("L3 news spill alpha optimization complete")
    print(f"  Metric: {result['metric']} (train_fraction={result['train_fraction']})")
    print(f"  Alpha: {result['l3_news_spill_alpha']}")
    samples = result.get("sample_sizes") or {}
    print(
        "  Samples: "
        f"eligible={samples.get('eligible')} "
        f"train={samples.get('train')} "
        f"val={samples.get('validation')}"
    )
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
        print("  Dry run — L3 spill alpha not written.")
    else:
        print(f"  Merged into {output}")


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
    events_output: Path,
    train_fraction: float,
    min_samples: int,
    horizon_days: int | None,
    dry_run: bool,
    weight_step: float,
    skip_news: bool = False,
    news_only: bool = False,
    skip_events: bool = False,
    events_only: bool = False,
    skip_l3: bool = False,
) -> dict:
    if skip_news and news_only:
        raise FusionOptimizationError("Cannot combine --skip-news and --news-only.")
    if skip_events and events_only:
        raise FusionOptimizationError("Cannot combine --skip-events and --events-only.")
    if news_only and events_only:
        raise FusionOptimizationError("Cannot combine --news-only and --events-only.")

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
            "events": None,
            "l3_spill": None,
            "fusion": None,
        }

    events_result: dict | None = None
    event_graph_params = default_event_graph_params()

    if not skip_events:
        events_result = optimize_event_graph(
            rows,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
            news_params=None if skip_news else news_params,
            train_fraction=train_fraction,
            min_samples=min_samples,
        )
        event_graph_params = {
            "propagation_decay": events_result["propagation_decay"],
            "max_hops": events_result["max_hops"],
            "score_divisor": events_result["score_divisor"],
            "edge_scale": events_result["edge_scale"],
            "theme_edges": events_result["theme_edges"],
            "event_specs": events_result["event_specs"],
        }
        if not dry_run:
            save_event_graph_params(events_result, path=events_output)

    if events_only:
        return {
            "news": news_result,
            "events": events_result,
            "l3_spill": None,
            "fusion": None,
        }

    l3_result: dict | None = None
    l3_alpha: float | None = None
    if not skip_l3:
        l3_result = optimize_l3_spill_alpha(
            rows,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
            news_params=None if skip_news else news_params,
            event_graph_params=None if skip_events else event_graph_params,
            train_fraction=train_fraction,
            min_samples=min_samples,
        )
        l3_alpha = float(l3_result["l3_news_spill_alpha"])
        if not dry_run:
            save_l3_spill_alpha(l3_alpha, path=news_output, result=l3_result)
    else:
        l3_alpha = get_l3_news_spill_alpha()

    # Re-sim only when news and/or events params were applied; else keep stored
    # news_score (already includes any spill from prediction time).
    if skip_news and skip_events:
        fusion_rows = rows
    else:
        fusion_rows = resimulate_fused_news_scores(
            rows,
            news_params=None if skip_news else news_params,
            event_graph_params=None if skip_events else event_graph_params,
            l3_spill_alpha=l3_alpha,
        )

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
        clear_event_graph_cache()

    return {
        "news": news_result,
        "events": events_result,
        "l3_spill": l3_result,
        "fusion": fusion_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune news scoring, L2 event-graph, L3 spill alpha, then fusion agent "
            "weights and horizon scales from evaluated predictions."
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
        "--events-output",
        type=Path,
        default=DEFAULT_EVENT_GRAPH_PATH,
        help="Path for tuned event-graph JSON (default: data/event_graph.json).",
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
        help=(
            "Minimum evaluated rows required (per horizon for fusion; "
            "with snapshots for news; with events for L2; with l3_news_spill for L3)."
        ),
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
        help="Skip news scoring phase.",
    )
    parser.add_argument(
        "--news-only",
        action="store_true",
        help="Tune and write news scoring only; skip event-graph, L3, and fusion phases.",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip L2 event-graph tuning phase.",
    )
    parser.add_argument(
        "--events-only",
        action="store_true",
        help="Tune event-graph (and news unless --skip-news); skip L3 and fusion phases.",
    )
    parser.add_argument(
        "--skip-l3",
        action="store_true",
        help="Skip L3 spill alpha tuning; keep file/env alpha for fusion resim.",
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
            events_output=args.events_output,
            train_fraction=args.train_fraction,
            min_samples=args.min_samples,
            horizon_days=args.horizon,
            dry_run=args.dry_run,
            weight_step=args.weight_step,
            skip_news=args.skip_news,
            news_only=args.news_only,
            skip_events=args.skip_events,
            events_only=args.events_only,
            skip_l3=args.skip_l3,
        )
    except (
        FusionOptimizationError,
        NewsScoringOptimizationError,
        EventGraphOptimizationError,
        L3SpillOptimizationError,
    ) as exc:
        raise SystemExit(str(exc)) from exc

    if result.get("news") is not None:
        _print_news_summary(result["news"], output=args.news_output, dry_run=args.dry_run)
    if result.get("events") is not None:
        _print_events_summary(
            result["events"], output=args.events_output, dry_run=args.dry_run
        )
    if result.get("l3_spill") is not None:
        _print_l3_summary(
            result["l3_spill"], output=args.news_output, dry_run=args.dry_run
        )
    if result.get("fusion") is not None:
        _print_fusion_summary(result["fusion"], output=args.output, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
