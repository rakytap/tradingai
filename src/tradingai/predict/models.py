from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AgentSignal(BaseModel):
    agent: str
    ticker: str | None = None
    directional_score: float = Field(ge=-1.0, le=1.0)
    bullish_score: float = Field(ge=0.0, le=1.0)
    bearish_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict)


class HorizonPrediction(BaseModel):
    horizon_days: int
    probability_up: float = Field(ge=0.0, le=1.0)
    expected_return_pct: float


class PredictionMetadata(BaseModel):
    sector: str | None = None
    regime: str | None = None
    catalyst_tags: list[str] = Field(default_factory=list)
    reason_tags: list[str] = Field(default_factory=list)
    prediction_type: str = "general_directional"


class PredictionReport(BaseModel):
    ticker: str
    agent_signals: list[AgentSignal]
    fusion: dict
    horizons: list[HorizonPrediction]
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: PredictionMetadata | None = None

    def to_public_dict(self) -> dict:
        payload = {
            "ticker": self.ticker,
            "agent_signals": [signal.model_dump() for signal in self.agent_signals],
            "fusion": self.fusion,
            "horizons": [horizon.model_dump() for horizon in self.horizons],
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
            "as_of": self.as_of.isoformat(),
        }
        if self.metadata is not None:
            payload["metadata"] = self.metadata.model_dump()
        return payload

