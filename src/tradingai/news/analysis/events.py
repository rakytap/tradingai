from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from tradingai.news.analysis.catalyst_extractor import _collect_text
from tradingai.news.models import AnalystAction, FilingItem, NewsItem
from tradingai.news.ontology import MAX_HOPS, PROPAGATION_DECAY, SECTOR_NODES, THEME_EDGES
from tradingai.regime.sector_proxies import normalize_sector_label


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


def _snippet(text: str, max_len: int = 160) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def extract_news_events_from_text(
    text: str,
    *,
    source: str | None = None,
    subject_ticker: str | None = None,
    confidence: float = 0.65,
) -> list[NewsEvent]:
    lowered = text.lower()
    events: list[NewsEvent] = []
    seen: set[str] = set()
    for event_type, polarity, themes, patterns in EVENT_SPECS:
        if event_type in seen:
            continue
        if any(re.search(expr, lowered) for expr in patterns):
            seen.add(event_type)
            events.append(
                NewsEvent(
                    event_type=event_type,
                    polarity=polarity,
                    themes=list(themes),
                    subject_ticker=subject_ticker.upper() if subject_ticker else None,
                    evidence=_snippet(text),
                    confidence=confidence,
                    source=source,
                )
            )
    return events


def extract_news_events(
    ticker: str,
    news: list[NewsItem],
    press_releases: list[NewsItem],
    filings: list[FilingItem],
    analyst_actions: list[AnalystAction],
    earnings_text: str | None = None,
    transcript_text: str | None = None,
) -> list[NewsEvent]:
    bundles = _collect_text(
        news,
        press_releases,
        filings,
        analyst_actions,
        earnings_text,
        transcript_text,
    )
    events: list[NewsEvent] = []
    seen_types: set[str] = set()
    for bundle in bundles:
        for event in extract_news_events_from_text(
            bundle.text,
            source=bundle.source,
            subject_ticker=ticker,
        ):
            if event.event_type in seen_types:
                continue
            seen_types.add(event.event_type)
            events.append(event)
    return events[:12]


def propagate_themes_to_sectors(
    themes: list[str],
    *,
    max_hops: int = MAX_HOPS,
    decay: float = PROPAGATION_DECAY,
) -> dict[str, float]:
    """Return sector -> accumulated absolute edge weight (sign applied later)."""
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

        if hop >= max_hops:
            continue

        for neighbor, edge_weight in THEME_EDGES.get(node, []):
            frontier.append((neighbor, weight * edge_weight * (decay if hop > 0 else 1.0), hop + 1))

    return sector_weights


def enrich_event_sectors(event: NewsEvent) -> NewsEvent:
    weights = propagate_themes_to_sectors(event.themes)
    # Keep sectors with non-trivial absolute weight.
    sectors = [
        sector
        for sector, weight in sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
        if abs(weight) >= 0.05
    ]
    event.sectors = sectors
    return event


def sector_exposures_from_events(events: list[NewsEvent]) -> dict[str, float]:
    exposures: dict[str, float] = {}
    for event in events:
        enrich_event_sectors(event)
        theme_sectors = propagate_themes_to_sectors(event.themes)
        for sector, weight in theme_sectors.items():
            contribution = float(event.polarity) * float(event.confidence) * weight
            exposures[sector] = exposures.get(sector, 0.0) + contribution
    return exposures


def score_events_for_ticker(
    events: list[NewsEvent],
    *,
    sector: str | None,
) -> dict:
    """Aggregate event exposure for a ticker's sector into [-1, 1] score."""
    if not events:
        return {
            "event_score": None,
            "propagated_sectors": {},
            "confidence": None,
            "reasons": [],
        }

    enriched = [enrich_event_sectors(event) for event in events]
    exposures = sector_exposures_from_events(enriched)
    canonical = normalize_sector_label(sector) or sector

    if canonical and canonical in exposures:
        raw_score = exposures[canonical]
    elif exposures:
        # No mapped sector: use mean absolute-weighted exposure across sectors.
        raw_score = sum(exposures.values()) / max(len(exposures), 1)
    else:
        raw_score = 0.0

    # Soft saturate.
    event_score = max(-1.0, min(1.0, raw_score / 1.5))
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
