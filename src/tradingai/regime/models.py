from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MacroFeatures(BaseModel):
    vix: float | None = None
    vix_percentile_1y: float | None = None
    spy_close: float | None = None
    spy_sma_50: float | None = None
    spy_sma_200: float | None = None
    spy_return_6m_pct: float | None = None
    treasury_10y: float | None = None
    treasury_2y: float | None = None
    yield_spread_2s10s_bps: float | None = None
    yield_curve_inverted: bool | None = None
    fed_funds_rate: float | None = None
    inflation_breakeven_10y: float | None = None
    cpi_yoy_pct: float | None = None
    credit_spread_hy_oas: float | None = None
    credit_spread_percentile_1y: float | None = None
    hyg_lqd_ratio: float | None = None
    sources_used: list[str] = Field(default_factory=list)


class MarketFactors(BaseModel):
    equity_trend: str
    vol: str
    risk: str
    rates: str
    credit: str
    growth_value: str
    ai_cycle: str | None = None
    geopolitics: str | None = None


class SectorStance(BaseModel):
    sector: str
    proxy: str
    regime: str
    stance: str


class RegimeState(BaseModel):
    market: MarketFactors
    sectors: list[SectorStance] = Field(default_factory=list)
    market_regime: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict = Field(default_factory=dict)
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def sector_stance_map(self) -> dict[str, str]:
        return {item.sector: item.stance for item in self.sectors}

    def stance_for_sector(self, sector: str | None) -> str | None:
        if not sector:
            return None
        mapping = self.sector_stance_map()
        if sector in mapping:
            return mapping[sector]
        lowered = sector.strip().lower()
        for name, stance in mapping.items():
            if name.lower() == lowered:
                return stance
        return None

    def to_public_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "confidence": round(self.confidence, 2),
            "market_regime": self.market_regime,
            "market": self.market.model_dump(),
            "sectors": [item.model_dump() for item in self.sectors],
            "evidence": self.evidence,
        }


class RegimeReport(BaseModel):
    regime: str
    recommended_strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    features: dict
    signals: list[dict]
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_public_dict(self) -> dict:
        return {
            "regime": self.regime,
            "recommended_strategy": self.recommended_strategy,
            "confidence": round(self.confidence, 2),
            "features": self.features,
            "signals": self.signals,
            "as_of": self.as_of.isoformat(),
        }
