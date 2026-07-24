from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from tradingai.news.analysis.catalyst_extractor import TextBundle, _collect_text
from tradingai.news.models import AnalystAction, FilingItem, NewsItem
from tradingai.news.ontology import MAX_HOPS, PROPAGATION_DECAY, SECTOR_NODES, THEME_EDGES
from tradingai.regime.sector_proxies import normalize_sector_label

EVENT_CORPUS_MAX_ITEMS = 40
EVENT_CORPUS_MAX_CHARS = 500
DEFAULT_SCORE_DIVISOR = 1.5
DEFAULT_EDGE_SCALE = 1.0


@dataclass
class NewsEvent:
    event_type: str
    polarity: int
    themes: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    subject_ticker: str | None = None
    evidence: str = ""
    confidence: float = 0.6
    source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# (event_type, polarity, seed_themes, patterns)
EVENT_SPECS: list[tuple[str, int, list[str], list[str]]] = [
    (
        "ai_capex_increase",
        1,
        ["ai_infra"],
        [
            r"ai capex",
            r"artificial intelligence (capex|spend|investment)",
            r"data.?center (capex|spend|investment|expansion)",
            r"gpu (demand|orders|shortage)",
            r"increas(e|es|ed|ing).{0,40}(ai|gpu|data.?center).{0,20}(spend|capex|investment)",
            r"(spend|capex|investment).{0,40}(ai|gpu|data.?center)",
        ],
    ),
    (
        "export_restriction",
        -1,
        ["semiconductors", "geopolitics"],
        [
            r"export (ban|restriction|control)",
            r"chip (ban|restriction)",
            r"semiconductor (ban|restriction|export)",
        ],
    ),
    (
        "earnings_beat",
        1,
        ["earnings_optimism"],
        [
            r"beat estimates",
            r"earnings surprise",
            r"raised guidance",
            r"raises guidance",
            r"topped expectations",
        ],
    ),
    (
        "earnings_miss",
        -1,
        ["earnings_optimism"],
        [
            r"missed expectations",
            r"miss(ed)? estimates",
            r"lowered guidance",
            r"cuts guidance",
            r"cut guidance",
        ],
    ),
    (
        "easing_expectation",
        1,
        ["rates_easing"],
        [
            r"rate cut",
            r"cuts rates",
            r"dovish",
            r"fed pivot",
            r"easing cycle",
        ],
    ),
    (
        "trade_shock",
        -1,
        ["geopolitics"],
        [
            r"\btariff",
            r"sanction",
            r"trade war",
            r"geopolitical",
        ],
    ),
    (
        "product_launch",
        1,
        ["product_cycle"],
        [
            r"product launch",
            r"unveiled",
            r"general availability",
            r"new product",
            r"\brollout\b",
        ],
    ),
    (
        "analyst_upgrade",
        1,
        ["earnings_optimism"],
        [
            r"\bupgrade[sd]?\b",
            r"initiated coverage",
            r"\boutperform\b",
            r"price target raised",
        ],
    ),
    (
        "analyst_downgrade",
        -1,
        ["earnings_optimism"],
        [
            r"\bdowngrade[sd]?\b",
            r"\bunderperform\b",
            r"price target cut",
            r"price target lowered",
        ],
    ),
    (
        "regulation_pressure",
        -1,
        ["regulation_risk"],
        [
            r"\bantitrust\b",
            r"regulatory (probe|investigation|scrutiny)",
            r"\bsec\b.{0,30}(charge|investigat)",
        ],
    ),
]


def default_theme_edges() -> dict[str, list[list]]:
    return {
        theme: [[neighbor, float(weight)] for neighbor, weight in edges]
        for theme, edges in THEME_EDGES.items()
    }


def default_event_specs() -> list[dict[str, Any]]:
    return [
        {
            "event_type": event_type,
            "enabled": True,
            "polarity": int(polarity),
            "themes": list(themes),
            "patterns": list(patterns),
        }
        for event_type, polarity, themes, patterns in EVENT_SPECS
    ]


def default_event_graph_params() -> dict[str, Any]:
    return {
        "propagation_decay": float(PROPAGATION_DECAY),
        "max_hops": int(MAX_HOPS),
        "score_divisor": float(DEFAULT_SCORE_DIVISOR),
        "edge_scale": float(DEFAULT_EDGE_SCALE),
        "theme_edges": default_theme_edges(),
        "event_specs": default_event_specs(),
    }


def theme_catalog() -> list[str]:
    return sorted(THEME_EDGES.keys())


def _runtime_graph_params(graph_params: dict[str, Any] | None = None) -> dict[str, Any]:
    if graph_params is not None:
        return graph_params
    from tradingai.news.config import get_event_graph_params

    return get_event_graph_params()


def _enabled_specs(graph_params: dict[str, Any]) -> list[dict[str, Any]]:
    specs = graph_params.get("event_specs") or []
    return [
        spec
        for spec in specs
        if isinstance(spec, dict)
        and spec.get("enabled", True)
        and spec.get("event_type")
        and spec.get("patterns")
    ]


def _theme_edges_map(graph_params: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    raw = graph_params.get("theme_edges") or {}
    scale = float(graph_params.get("edge_scale", DEFAULT_EDGE_SCALE))
    result: dict[str, list[tuple[str, float]]] = {}
    if not isinstance(raw, dict):
        return result
    for theme, edges in raw.items():
        parsed: list[tuple[str, float]] = []
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            neighbor = str(edge[0])
            try:
                weight = float(edge[1]) * scale
            except (TypeError, ValueError):
                continue
            if weight == 0.0:
                continue
            parsed.append((neighbor, weight))
        result[str(theme)] = parsed
    return result


def _snippet(text: str, max_len: int = 160) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def build_event_text_corpus(
    bundles: list[TextBundle],
    *,
    max_items: int = EVENT_CORPUS_MAX_ITEMS,
    max_chars: int = EVENT_CORPUS_MAX_CHARS,
) -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []
    for bundle in bundles[:max_items]:
        cleaned = " ".join((bundle.text or "").split())
        if not cleaned:
            continue
        if len(cleaned) > max_chars:
            cleaned = cleaned[: max_chars - 1] + "…"
        corpus.append({"text": cleaned, "source": bundle.source or "unknown"})
    return corpus


def extract_news_events_from_text(
    text: str,
    *,
    source: str | None = None,
    subject_ticker: str | None = None,
    confidence: float = 0.65,
    graph_params: dict[str, Any] | None = None,
) -> list[NewsEvent]:
    params = _runtime_graph_params(graph_params)
    lowered = text.lower()
    events: list[NewsEvent] = []
    seen: set[str] = set()
    for spec in _enabled_specs(params):
        event_type = str(spec["event_type"])
        if event_type in seen:
            continue
        patterns = [str(p) for p in (spec.get("patterns") or []) if p]
        if not patterns:
            continue
        if any(re.search(expr, lowered) for expr in patterns):
            seen.add(event_type)
            themes = [str(t) for t in (spec.get("themes") or [])]
            polarity = int(spec.get("polarity", 1))
            events.append(
                NewsEvent(
                    event_type=event_type,
                    polarity=polarity,
                    themes=themes,
                    subject_ticker=subject_ticker.upper() if subject_ticker else None,
                    evidence=_snippet(text),
                    confidence=confidence,
                    source=source,
                )
            )
    return events


def extract_news_events_from_corpus(
    corpus: list[dict[str, Any]],
    *,
    subject_ticker: str | None = None,
    graph_params: dict[str, Any] | None = None,
    max_events: int = 12,
) -> list[NewsEvent]:
    events: list[NewsEvent] = []
    seen_types: set[str] = set()
    for item in corpus:
        text = item.get("text") if isinstance(item, dict) else None
        if not text:
            continue
        source = item.get("source") if isinstance(item, dict) else None
        for event in extract_news_events_from_text(
            str(text),
            source=str(source) if source else None,
            subject_ticker=subject_ticker,
            graph_params=graph_params,
        ):
            if event.event_type in seen_types:
                continue
            seen_types.add(event.event_type)
            events.append(event)
            if len(events) >= max_events:
                return events
    return events


def extract_news_events(
    ticker: str,
    news: list[NewsItem],
    press_releases: list[NewsItem],
    filings: list[FilingItem],
    analyst_actions: list[AnalystAction],
    earnings_text: str | None = None,
    transcript_text: str | None = None,
    *,
    graph_params: dict[str, Any] | None = None,
) -> list[NewsEvent]:
    bundles = _collect_text(
        news,
        press_releases,
        filings,
        analyst_actions,
        earnings_text,
        transcript_text,
    )
    corpus = build_event_text_corpus(bundles)
    return extract_news_events_from_corpus(
        corpus,
        subject_ticker=ticker,
        graph_params=graph_params,
    )


def collect_event_extraction_inputs(
    ticker: str,
    news: list[NewsItem],
    press_releases: list[NewsItem],
    filings: list[FilingItem],
    analyst_actions: list[AnalystAction],
    earnings_text: str | None = None,
    transcript_text: str | None = None,
    *,
    graph_params: dict[str, Any] | None = None,
) -> tuple[list[NewsEvent], list[dict[str, str]]]:
    """Return (events, corpus) from the same text bundles used for extraction."""
    bundles = _collect_text(
        news,
        press_releases,
        filings,
        analyst_actions,
        earnings_text,
        transcript_text,
    )
    corpus = build_event_text_corpus(bundles)
    events = extract_news_events_from_corpus(
        corpus,
        subject_ticker=ticker,
        graph_params=graph_params,
    )
    return events, corpus


def propagate_themes_to_sectors(
    themes: list[str],
    *,
    max_hops: int | None = None,
    decay: float | None = None,
    graph_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return sector -> accumulated absolute edge weight (sign applied later)."""
    params = _runtime_graph_params(graph_params)
    hops = int(max_hops if max_hops is not None else params.get("max_hops", MAX_HOPS))
    prop_decay = float(decay if decay is not None else params.get("propagation_decay", PROPAGATION_DECAY))
    theme_edges = _theme_edges_map(params)

    sector_weights: dict[str, float] = {}
    frontier: list[tuple[str, float, int]] = [(theme, 1.0, 0) for theme in themes]
    visited: set[tuple[str, int]] = set()

    while frontier:
        node, weight, hop = frontier.pop(0)
        key = (node, hop)
        if key in visited:
            continue
        visited.add(key)

        if node in SECTOR_NODES:
            sector_weights[node] = sector_weights.get(node, 0.0) + weight
            continue

        if hop >= hops:
            continue

        for neighbor, edge_weight in theme_edges.get(node, []):
            frontier.append(
                (neighbor, weight * edge_weight * (prop_decay if hop > 0 else 1.0), hop + 1)
            )

    return sector_weights


def enrich_event_sectors(
    event: NewsEvent,
    *,
    graph_params: dict[str, Any] | None = None,
) -> NewsEvent:
    weights = propagate_themes_to_sectors(event.themes, graph_params=graph_params)
    sectors = [
        sector
        for sector, weight in sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
        if abs(weight) >= 0.05
    ]
    event.sectors = sectors
    return event


def sector_exposures_from_events(
    events: list[NewsEvent],
    *,
    graph_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for event in events:
        enrich_event_sectors(event, graph_params=graph_params)
        theme_sectors = propagate_themes_to_sectors(event.themes, graph_params=graph_params)
        for sector, weight in theme_sectors.items():
            contribution = float(event.polarity) * float(event.confidence) * weight
            exposures[sector] = exposures.get(sector, 0.0) + contribution
    return exposures


def events_from_payload(
    raw_events: list[dict[str, Any]] | None,
    *,
    graph_params: dict[str, Any] | None = None,
) -> list[NewsEvent]:
    """Rebuild NewsEvent list from stored payload, applying enabled/polarity/themes from config."""
    params = _runtime_graph_params(graph_params)
    spec_by_type = {
        str(spec["event_type"]): spec
        for spec in (params.get("event_specs") or [])
        if isinstance(spec, dict) and spec.get("event_type")
    }
    events: list[NewsEvent] = []
    for raw in raw_events or []:
        if not isinstance(raw, dict):
            continue
        event_type = str(raw.get("event_type") or "")
        if not event_type:
            continue
        spec = spec_by_type.get(event_type)
        if spec is not None and not spec.get("enabled", True):
            continue
        if spec is not None:
            polarity = int(spec.get("polarity", raw.get("polarity", 1)))
            themes = [str(t) for t in (spec.get("themes") or raw.get("themes") or [])]
        else:
            polarity = int(raw.get("polarity", 1))
            themes = [str(t) for t in (raw.get("themes") or [])]
        events.append(
            NewsEvent(
                event_type=event_type,
                polarity=polarity,
                themes=themes,
                sectors=list(raw.get("sectors") or []),
                subject_ticker=raw.get("subject_ticker"),
                evidence=str(raw.get("evidence") or ""),
                confidence=float(raw.get("confidence", 0.6)),
                source=raw.get("source"),
            )
        )
    return events


def score_events_for_ticker(
    events: list[NewsEvent],
    *,
    sector: str | None,
    graph_params: dict[str, Any] | None = None,
) -> dict:
    """Aggregate event exposure for a ticker's sector into [-1, 1] score."""
    params = _runtime_graph_params(graph_params)
    if not events:
        return {
            "event_score": None,
            "propagated_sectors": {},
            "confidence": None,
            "reasons": [],
            "events": [],
        }

    enriched = [enrich_event_sectors(event, graph_params=params) for event in events]
    exposures = sector_exposures_from_events(enriched, graph_params=params)
    canonical = normalize_sector_label(sector) or sector
    divisor = float(params.get("score_divisor", DEFAULT_SCORE_DIVISOR)) or DEFAULT_SCORE_DIVISOR

    if canonical and canonical in exposures:
        raw_score = exposures[canonical]
    elif exposures:
        raw_score = sum(exposures.values()) / max(len(exposures), 1)
    else:
        raw_score = 0.0

    event_score = max(-1.0, min(1.0, raw_score / divisor))
    confidences = [event.confidence for event in enriched]
    confidence = max(0.4, sum(confidences) / len(confidences))

    reasons: list[str] = []
    for event in enriched[:3]:
        chain = " → ".join([*event.themes[:2], *(event.sectors[:1] or [])])
        reasons.append(f"{event.event_type} ({event.polarity:+d}): {chain}")
        if event.evidence:
            reasons.append(event.evidence)

    return {
        "event_score": round(event_score, 4),
        "propagated_sectors": {key: round(value, 4) for key, value in exposures.items()},
        "confidence": round(confidence, 4),
        "reasons": reasons[:6],
        "events": [event.to_dict() for event in enriched],
    }
