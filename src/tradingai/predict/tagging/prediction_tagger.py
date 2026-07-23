from __future__ import annotations

import re

from tradingai.news.analysis.catalyst_extractor import CATALYST_PATTERNS, RISK_PATTERNS
from tradingai.predict.config import get_ml_horizon_days
from tradingai.predict.models import AgentSignal, PredictionMetadata, PredictionReport


def extract_catalyst_keys_from_text(text: str) -> list[str]:
    lowered = text.lower()
    keys: list[str] = []
    for label, expressions in CATALYST_PATTERNS.items():
        if any(re.search(expr, lowered) for expr in expressions):
            keys.append(label)
    return keys


def extract_reason_tags(news_raw: dict) -> list[str]:
    tags: list[str] = []
    for risk in news_raw.get("risk_factors", []):
        normalized = str(risk).strip().lower().replace(" ", "_")
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags[:5]


def extract_catalyst_tags(agent_signals: list[AgentSignal]) -> list[str]:
    news_signal = next((signal for signal in agent_signals if signal.agent == "news"), None)
    if not news_signal:
        return []

    tags: list[str] = []
    raw = news_signal.raw
    for catalyst in raw.get("catalysts", []):
        for key in extract_catalyst_keys_from_text(str(catalyst)):
            if key not in tags:
                tags.append(key)

    headline_blob = " ".join(
        str(value)
        for value in [
            *raw.get("catalysts", []),
            *raw.get("risk_factors", []),
        ]
    )
    for key in extract_catalyst_keys_from_text(headline_blob):
        if key not in tags:
            tags.append(key)
    return tags


def classify_prediction_type(
    report: PredictionReport,
    metadata: PredictionMetadata,
    primary_horizon_prob: float,
) -> str:
    technical = next(
        (signal.directional_score for signal in report.agent_signals if signal.agent == "technical"),
        0.0,
    )
    news_score = report.fusion.get("news_score", 0.0)
    combined = report.fusion.get("combined_score", 0.0)
    tags = set(metadata.catalyst_tags)

    if "earnings_surprise" in tags and primary_horizon_prob >= 0.6 and technical > 0:
        return "earnings_driven_rally"
    if "product_launch" in tags and primary_horizon_prob >= 0.55:
        return "product_catalyst_rally"
    if "analyst_action" in tags and news_score > 0:
        return "analyst_upgrade_rally"
    if metadata.regime in {"bear", "bearish", "market:bear"} and combined < 0:
        return "risk_off_drawdown"
    return "general_directional"


def _primary_horizon_prob(report: PredictionReport) -> float:
    target = get_ml_horizon_days()
    for horizon in report.horizons:
        if horizon.horizon_days == target:
            return horizon.probability_up
    return report.horizons[0].probability_up if report.horizons else 0.5


def build_prediction_metadata(
    report: PredictionReport,
    *,
    sector: str | None,
) -> PredictionMetadata:
    regime_signal = next(
        (signal for signal in report.agent_signals if signal.agent == "regime"),
        None,
    )
    news_signal = next(
        (signal for signal in report.agent_signals if signal.agent == "news"),
        None,
    )
    catalyst_tags = extract_catalyst_tags(report.agent_signals)
    reason_tags = extract_reason_tags(news_signal.raw if news_signal else {})
    regime = None
    if regime_signal:
        regime = regime_signal.raw.get("regime")

    primary_prob = _primary_horizon_prob(report)
    metadata = PredictionMetadata(
        sector=sector,
        regime=regime,
        catalyst_tags=catalyst_tags,
        reason_tags=reason_tags,
        prediction_type="general_directional",
    )
    metadata.prediction_type = classify_prediction_type(report, metadata, primary_prob)
    return metadata
