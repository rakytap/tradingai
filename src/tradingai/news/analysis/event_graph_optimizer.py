from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from tradingai.news.analysis.events import (
    default_event_graph_params,
    extract_news_events_from_corpus,
    events_from_payload,
    score_events_for_ticker,
    theme_catalog,
)
from tradingai.news.analysis.scoring_optimizer import (
    _combined_probability,
    _parse_agent_payload,
    extract_scoring_snapshot,
    sentiment_to_directional,
)
from tradingai.news.analysis.sentiment_scorer import rescore_from_snapshot
from tradingai.predict.fusion.optimizer import _metrics_for_rows, chronological_split

DECAY_CANDIDATES = (0.3, 0.5, 0.7)
DIVISOR_CANDIDATES = (1.0, 1.5, 2.0)
EDGE_SCALE_CANDIDATES = (0.75, 1.0, 1.25)
EDGE_WEIGHT_CANDIDATES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class EventGraphOptimizationError(ValueError):
    """Raised when L2 event-graph params cannot be tuned."""


def extract_news_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_agent_payload(row)
    news = payload.get("news")
    if isinstance(news, dict):
        return news
    return payload if isinstance(payload, dict) else {}


def extract_event_text_corpus(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    news = extract_news_payload(row)
    corpus = news.get("event_text_corpus")
    if isinstance(corpus, list) and corpus:
        return [item for item in corpus if isinstance(item, dict) and item.get("text")]
    return None


def extract_stored_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    news = extract_news_payload(row)
    events = news.get("events")
    if isinstance(events, list):
        return [e for e in events if isinstance(e, dict)]
    return []


def row_sector(row: dict[str, Any]) -> str | None:
    sector = row.get("sector")
    return str(sector) if sector else None


def rows_with_event_signal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if extract_event_text_corpus(row) or extract_stored_events(row):
            eligible.append(row)
    return eligible


def resolve_events_for_row(
    row: dict[str, Any],
    graph_params: dict[str, Any],
) -> list:
    corpus = extract_event_text_corpus(row)
    ticker = str(row.get("ticker") or "")
    if corpus:
        return extract_news_events_from_corpus(
            corpus,
            subject_ticker=ticker or None,
            graph_params=graph_params,
        )
    return events_from_payload(extract_stored_events(row), graph_params=graph_params)


def event_directional_for_row(
    row: dict[str, Any],
    graph_params: dict[str, Any],
) -> float | None:
    events = resolve_events_for_row(row, graph_params)
    if not events:
        return None
    scored = score_events_for_ticker(
        events,
        sector=row_sector(row) or row.get("sector"),
        graph_params=graph_params,
    )
    score = scored.get("event_score")
    return float(score) if score is not None else None


def sentiment_directional_for_row(
    row: dict[str, Any],
    news_params: dict[str, Any] | None,
) -> float | None:
    if news_params is None:
        return None
    snapshot = extract_scoring_snapshot(row)
    if snapshot is None:
        return None
    from tradingai.news.analysis.scoring_optimizer import _as_of_for_row

    sentiment = rescore_from_snapshot(snapshot, news_params, as_of=_as_of_for_row(row))
    return sentiment_to_directional(sentiment)


def resolve_news_score_for_row(
    row: dict[str, Any],
    *,
    news_params: dict[str, Any] | None = None,
    event_graph_params: dict[str, Any] | None = None,
) -> float | None:
    """Prefer event score when resolvable; else sentiment; else None (keep original)."""
    if event_graph_params is not None:
        event_score = event_directional_for_row(row, event_graph_params)
        if event_score is not None:
            return round(event_score, 4)
    sentiment_score = sentiment_directional_for_row(row, news_params)
    if sentiment_score is not None:
        return round(sentiment_score, 4)
    return None


def resimulate_fused_news_scores(
    rows: list[dict[str, Any]],
    *,
    news_params: dict[str, Any] | None = None,
    event_graph_params: dict[str, Any] | None = None,
    l3_spill_alpha: float | None = None,
) -> list[dict[str, Any]]:
    from tradingai.news.analysis.l3_spillover import (
        apply_l3_spill_to_score,
        extract_l3_spill_inputs,
    )

    updated: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        resolved = resolve_news_score_for_row(
            row,
            news_params=news_params,
            event_graph_params=event_graph_params,
        )
        if resolved is not None:
            news_score = float(resolved)
            if l3_spill_alpha is not None:
                inputs = extract_l3_spill_inputs(row)
                if inputs is not None:
                    news_score = apply_l3_spill_to_score(
                        news_score, inputs, l3_spill_alpha
                    )
            copy["news_score"] = round(news_score, 4)
        updated.append(copy)
    return updated


def _brier_for_event_params(
    rows: list[dict[str, Any]],
    graph_params: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> float:
    if not rows:
        raise EventGraphOptimizationError("No rows available for event-graph Brier scoring")
    total = 0.0
    for row in rows:
        news_score = resolve_news_score_for_row(
            row,
            news_params=news_params,
            event_graph_params=graph_params,
        )
        if news_score is None:
            news_score = float(row.get("news_score") or 0.0)
        prob = _combined_probability(
            row,
            news_score=news_score,
            fusion_weights=fusion_weights,
            horizon_scales=horizon_scales,
        )
        actual_up = 1.0 if float(row["actual_return_pct"]) > 0 else 0.0
        total += (prob - actual_up) ** 2
    return total / len(rows)


def _metrics_with_event_params(
    rows: list[dict[str, Any]],
    graph_params: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> dict[str, float | None]:
    simulated = resimulate_fused_news_scores(
        rows,
        news_params=news_params,
        event_graph_params=graph_params,
    )
    return _metrics_for_rows(simulated, fusion_weights, horizon_scales)


def _serialize_graph_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "propagation_decay": round(float(params["propagation_decay"]), 4),
        "max_hops": int(params["max_hops"]),
        "score_divisor": round(float(params["score_divisor"]), 4),
        "edge_scale": round(float(params["edge_scale"]), 4),
        "theme_edges": {
            theme: [[neighbor, round(float(weight), 4)] for neighbor, weight in edges]
            for theme, edges in params["theme_edges"].items()
        },
        "event_specs": deepcopy(params["event_specs"]),
    }


def _edge_list(params: dict[str, Any]) -> list[tuple[str, int, str, float]]:
    edges: list[tuple[str, int, str, float]] = []
    for theme, neighbors in params["theme_edges"].items():
        for idx, edge in enumerate(neighbors):
            neighbor, weight = edge[0], float(edge[1])
            edges.append((theme, idx, str(neighbor), weight))
    return edges


def _set_edge_weight(
    params: dict[str, Any],
    theme: str,
    idx: int,
    weight: float,
) -> None:
    params["theme_edges"][theme][idx][1] = float(weight)


def _candidate_edge_weights(baseline: float) -> list[float]:
    if baseline < 0:
        return sorted({-w for w in EDGE_WEIGHT_CANDIDATES}, reverse=True)
    return list(EDGE_WEIGHT_CANDIDATES)


def _optimize_knobs(
    train_rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> dict[str, Any]:
    best = deepcopy(params)
    best_brier = _brier_for_event_params(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    for decay in DECAY_CANDIDATES:
        for divisor in DIVISOR_CANDIDATES:
            for edge_scale in EDGE_SCALE_CANDIDATES:
                candidate = deepcopy(best)
                candidate["propagation_decay"] = decay
                candidate["score_divisor"] = divisor
                candidate["edge_scale"] = edge_scale
                brier = _brier_for_event_params(
                    train_rows,
                    candidate,
                    news_params=news_params,
                    fusion_weights=fusion_weights,
                    horizon_scales=horizon_scales,
                )
                if brier < best_brier:
                    best_brier = brier
                    best = candidate
    return best


def _optimize_theme_edges(
    train_rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
    passes: int = 2,
) -> dict[str, Any]:
    best = deepcopy(params)
    best_brier = _brier_for_event_params(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    for _ in range(passes):
        improved = False
        for theme, idx, _neighbor, baseline_weight in _edge_list(best):
            for weight in _candidate_edge_weights(baseline_weight):
                if abs(weight - float(best["theme_edges"][theme][idx][1])) < 1e-12:
                    continue
                candidate = deepcopy(best)
                _set_edge_weight(candidate, theme, idx, weight)
                brier = _brier_for_event_params(
                    train_rows,
                    candidate,
                    news_params=news_params,
                    fusion_weights=fusion_weights,
                    horizon_scales=horizon_scales,
                )
                if brier < best_brier:
                    best_brier = brier
                    best = candidate
                    improved = True
        if not improved:
            break
    return best


def _optimize_event_enablement(
    train_rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> dict[str, Any]:
    best = deepcopy(params)
    best_brier = _brier_for_event_params(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    for idx, spec in enumerate(best["event_specs"]):
        for enabled in (True, False):
            if bool(spec.get("enabled", True)) is enabled:
                continue
            candidate = deepcopy(best)
            candidate["event_specs"][idx]["enabled"] = enabled
            brier = _brier_for_event_params(
                train_rows,
                candidate,
                news_params=news_params,
                fusion_weights=fusion_weights,
                horizon_scales=horizon_scales,
            )
            if brier < best_brier:
                best_brier = brier
                best = candidate
    return best


def _theme_remap_candidates(current: list[str]) -> list[list[str]]:
    catalog = theme_catalog()
    options: list[list[str]] = [list(current)]
    for theme in catalog:
        singleton = [theme]
        if singleton not in options:
            options.append(singleton)
    if len(current) == 1:
        for theme in catalog:
            pair = sorted({current[0], theme})
            if len(pair) == 2 and pair not in options:
                options.append(pair)
    return options


def _optimize_theme_remap(
    train_rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> dict[str, Any]:
    best = deepcopy(params)
    best_brier = _brier_for_event_params(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    for idx, spec in enumerate(best["event_specs"]):
        if not spec.get("enabled", True):
            continue
        current = [str(t) for t in (spec.get("themes") or [])]
        for themes in _theme_remap_candidates(current):
            if themes == current:
                continue
            candidate = deepcopy(best)
            candidate["event_specs"][idx]["themes"] = themes
            brier = _brier_for_event_params(
                train_rows,
                candidate,
                news_params=news_params,
                fusion_weights=fusion_weights,
                horizon_scales=horizon_scales,
            )
            if brier < best_brier:
                best_brier = brier
                best = candidate
                current = themes
    return best


def _optimize_pattern_ablation(
    train_rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    news_params: dict[str, Any] | None,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
) -> dict[str, Any]:
    corpus_rows = [row for row in train_rows if extract_event_text_corpus(row)]
    if not corpus_rows:
        return params

    best = deepcopy(params)
    best_brier = _brier_for_event_params(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    improved = True
    while improved:
        improved = False
        for spec_idx, spec in enumerate(best["event_specs"]):
            if not spec.get("enabled", True):
                continue
            patterns = list(spec.get("patterns") or [])
            if len(patterns) <= 1:
                continue
            for pat_idx in range(len(patterns)):
                candidate = deepcopy(best)
                new_patterns = [
                    p for i, p in enumerate(candidate["event_specs"][spec_idx]["patterns"]) if i != pat_idx
                ]
                if not new_patterns:
                    continue
                candidate["event_specs"][spec_idx]["patterns"] = new_patterns
                brier = _brier_for_event_params(
                    train_rows,
                    candidate,
                    news_params=news_params,
                    fusion_weights=fusion_weights,
                    horizon_scales=horizon_scales,
                )
                if brier < best_brier:
                    best_brier = brier
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    return best


def optimize_event_graph(
    rows: list[dict[str, Any]],
    *,
    fusion_weights: tuple[float, float, float],
    horizon_scales: dict[int, float],
    news_params: dict[str, Any] | None = None,
    train_fraction: float = 0.7,
    min_samples: int = 30,
) -> dict[str, Any]:
    eligible = rows_with_event_signal(rows)
    if len(eligible) < min_samples:
        raise EventGraphOptimizationError(
            f"Only {len(eligible)} evaluated rows have events or event_text_corpus "
            f"(need at least {min_samples}). "
            "Generate predictions after event payload support, run tradingai-predict-eval, "
            "then re-run tradingai-tune-fusion."
        )

    required = ("technical_score", "regime_score", "actual_return_pct", "horizon_days")
    eligible = [
        row
        for row in eligible
        if all(row.get(key) is not None for key in required)
    ]
    if len(eligible) < min_samples:
        raise EventGraphOptimizationError(
            f"Only {len(eligible)} event rows have complete fusion component scores "
            f"(need at least {min_samples})."
        )

    train_rows, val_rows = chronological_split(eligible, train_fraction=train_fraction)
    if not val_rows:
        raise EventGraphOptimizationError(
            "No validation rows after chronological split for event-graph tuning. "
            "Collect more evaluated predictions or lower --train-fraction."
        )

    baseline = default_event_graph_params()
    best = deepcopy(baseline)
    best = _optimize_knobs(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    best = _optimize_theme_edges(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    best = _optimize_event_enablement(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    best = _optimize_theme_remap(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    best = _optimize_pattern_ablation(
        train_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )

    serialized = _serialize_graph_params(best)
    baseline_metrics = _metrics_with_event_params(
        val_rows,
        baseline,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    tuned_metrics = _metrics_with_event_params(
        val_rows,
        best,
        news_params=news_params,
        fusion_weights=fusion_weights,
        horizon_scales=horizon_scales,
    )
    corpus_count = sum(1 for row in eligible if extract_event_text_corpus(row))

    return {
        "tuned_at": datetime.now().astimezone().isoformat(),
        "metric": "brier",
        "train_fraction": train_fraction,
        **serialized,
        "sample_sizes": {
            "eligible": len(eligible),
            "with_event_text_corpus": corpus_count,
            "train": len(train_rows),
            "validation": len(val_rows),
        },
        "baseline_validation": baseline_metrics,
        "tuned_validation": tuned_metrics,
    }
