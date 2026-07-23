from __future__ import annotations

# Theme/sector edge weight: (neighbor_node, weight).
# Sector nodes use Step-1 board labels (Technology, Energy, ...).
THEME_EDGES: dict[str, list[tuple[str, float]]] = {
    "ai_infra": [
        ("semiconductors", 0.9),
        ("power_equipment", 0.6),
        ("Utilities", 0.4),
        ("Technology", 0.8),
    ],
    "semiconductors": [
        ("Technology", 1.0),
    ],
    "power_equipment": [
        ("Industrials", 0.7),
        ("Utilities", 0.5),
    ],
    "earnings_optimism": [
        ("Technology", 0.3),
        ("Financial Services", 0.2),
        ("Consumer Cyclical", 0.2),
    ],
    "rates_easing": [
        ("Real Estate", 0.6),
        ("Utilities", 0.4),
        ("Financial Services", -0.3),
    ],
    "geopolitics": [
        ("Technology", -0.4),
        ("Energy", 0.2),
    ],
    "product_cycle": [
        ("Technology", 0.5),
        ("Consumer Cyclical", 0.4),
    ],
    "regulation_risk": [
        ("Technology", -0.3),
        ("Communication Services", -0.3),
        ("Healthcare", -0.2),
    ],
}

SECTOR_NODES = {
    "Technology",
    "Energy",
    "Healthcare",
    "Financial Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Industrials",
    "Basic Materials",
    "Real Estate",
    "Utilities",
    "Communication Services",
}

PROPAGATION_DECAY = 0.5
MAX_HOPS = 2
