from __future__ import annotations

import re
from dataclasses import dataclass

from tradingai.news.models import AnalystAction, FilingItem, NewsItem

CATALYST_PATTERNS: dict[str, list[str]] = {
    "earnings_surprise": [
        r"beat estimates",
        r"missed expectations",
        r"earnings surprise",
        r"eps",
        r"revenue growth",
        r"raised guidance",
        r"lowered guidance",
    ],
    "lawsuit": [
        r"lawsuit",
        r"litigation",
        r"class action",
        r"settlement",
    ],
    "regulation": [
        r"\bsec\b",
        r"\bftc\b",
        r"antitrust",
        r"regulatory",
        r"compliance",
    ],
    "product_launch": [
        r"launch",
        r"unveiled",
        r"new product",
        r"rollout",
        r"general availability",
    ],
    "analyst_action": [
        r"upgrade",
        r"downgrade",
        r"price target",
        r"initiated coverage",
        r"outperform",
        r"underperform",
    ],
    "geopolitical": [
        r"tariff",
        r"export restriction",
        r"sanction",
        r"geopolitical",
        r"china",
        r"taiwan",
    ],
}

RISK_PATTERNS: dict[str, list[str]] = {
    "valuation": [r"valuation", r"overvalued", r"expensive", r"premium multiple"],
    "regulatory": [r"regulatory risk", r"investigation", r"antitrust", r"compliance"],
    "litigation": [r"lawsuit", r"litigation", r"legal risk"],
    "macro": [r"recession", r"inflation", r"interest rates", r"macro"],
    "supply_chain": [r"supply chain", r"shortage", r"inventory"],
    "competition": [r"competition", r"market share loss", r"rival"],
}

CATALYST_LABELS = {
    "earnings_surprise": "Earnings momentum or guidance update",
    "lawsuit": "Legal or litigation development",
    "regulation": "Regulatory or compliance development",
    "product_launch": "Product launch or rollout catalyst",
    "analyst_action": "Analyst rating or price target change",
    "geopolitical": "Geopolitical or trade exposure",
}


@dataclass
class TextBundle:
    text: str
    source: str


def _collect_text(
    news: list[NewsItem],
    press_releases: list[NewsItem],
    filings: list[FilingItem],
    analyst_actions: list[AnalystAction],
    earnings_text: str | None = None,
    transcript_text: str | None = None,
) -> list[TextBundle]:
    bundles: list[TextBundle] = []
    for item in news + press_releases:
        parts = [item.title, item.summary or ""]
        bundles.append(TextBundle(text=" ".join(parts), source=item.provider))
    for filing in filings:
        parts = [filing.title or "", filing.excerpt or ""]
        bundles.append(TextBundle(text=" ".join(parts), source=filing.provider))
    for action in analyst_actions:
        parts = [
            action.action,
            action.from_grade or "",
            action.to_grade or "",
            action.firm or "",
        ]
        bundles.append(TextBundle(text=" ".join(parts), source=action.provider))
    if earnings_text:
        bundles.append(TextBundle(text=earnings_text, source="sec_edgar"))
    if transcript_text:
        bundles.append(TextBundle(text=transcript_text, source="sec_edgar"))
    return bundles


def _match_patterns(text: str, patterns: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    for label, expressions in patterns.items():
        if any(re.search(expr, lowered) for expr in expressions):
            matches.append(label)
    return matches


def extract_catalysts(
    news: list[NewsItem],
    press_releases: list[NewsItem],
    filings: list[FilingItem],
    analyst_actions: list[AnalystAction],
    earnings_text: str | None = None,
    transcript_text: str | None = None,
) -> list[str]:
    bundles = _collect_text(
        news,
        press_releases,
        filings,
        analyst_actions,
        earnings_text,
        transcript_text,
    )
    found: list[str] = []
    seen: set[str] = set()
    for bundle in bundles:
        for label in _match_patterns(bundle.text, CATALYST_PATTERNS):
            phrase = CATALYST_LABELS.get(label, label.replace("_", " ").title())
            if phrase not in seen:
                seen.add(phrase)
                found.append(phrase)
    for action in analyst_actions:
        action_lower = action.action.lower()
        if "upgrade" in action_lower or "initiated" in action_lower:
            phrase = f"Analyst upgrade from {action.firm or 'a brokerage firm'}"
        elif "downgrade" in action_lower:
            phrase = f"Analyst downgrade from {action.firm or 'a brokerage firm'}"
        else:
            continue
        if phrase not in seen:
            seen.add(phrase)
            found.append(phrase)
    return found[:8]


def extract_risk_factors(
    news: list[NewsItem],
    press_releases: list[NewsItem],
    filings: list[FilingItem],
    earnings_text: str | None = None,
    transcript_text: str | None = None,
) -> list[str]:
    bundles = _collect_text(news, press_releases, filings, [], earnings_text, transcript_text)
    found: list[str] = []
    seen: set[str] = set()
    for bundle in bundles:
        for label in _match_patterns(bundle.text, RISK_PATTERNS):
            if label not in seen:
                seen.add(label)
                found.append(label.replace("_", " "))
    return found[:6]
