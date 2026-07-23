from __future__ import annotations

from tradingai.regime.analysis.classifier import classify_regime
from tradingai.regime.models import MacroFeatures, SectorStance
from tradingai.regime.sector_proxies import SECTOR_BOARD


def regime_to_stance(regime: str) -> str:
    if regime in {"bull", "bull_high_volatility"}:
        return "bullish"
    if regime in {"bear", "high_volatility_bearish"}:
        return "bearish"
    return "neutral"


def classify_sector_board(
    features_by_proxy: dict[str, MacroFeatures],
) -> list[SectorStance]:
    """Build L4 sector stances from per-proxy MacroFeatures."""
    board: list[SectorStance] = []
    for sector, proxy in SECTOR_BOARD.items():
        features = features_by_proxy.get(proxy)
        if features is None:
            continue
        regime = classify_regime(features)
        board.append(
            SectorStance(
                sector=sector,
                proxy=proxy,
                regime=regime,
                stance=regime_to_stance(regime),
            )
        )
    return board
