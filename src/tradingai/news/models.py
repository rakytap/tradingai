from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    title: str
    summary: str | None = None
    url: str | None = None
    source: str
    published_at: datetime | None = None
    provider: str


class SocialPost(BaseModel):
    title: str
    summary: str | None = None
    url: str | None = None
    platform: str
    source: str
    published_at: datetime | None = None
    engagement_score: float | None = None
    provider: str


class AnalystAction(BaseModel):
    firm: str | None = None
    action: str
    from_grade: str | None = None
    to_grade: str | None = None
    published_at: datetime | None = None
    provider: str


class FilingItem(BaseModel):
    form: str
    filed_at: datetime | None = None
    accession_number: str | None = None
    title: str | None = None
    excerpt: str | None = None
    url: str | None = None
    provider: str


class EarningsMaterial(BaseModel):
    form: str = "8-K"
    filed_at: datetime | None = None
    press_release: str | None = None
    transcript: str | None = None
    provider: str


class SentimentReport(BaseModel):
    ticker: str
    sentiment: float = Field(ge=0.0, le=1.0)
    catalysts: list[str]
    risk_factors: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)
    scoring_snapshot: dict | None = None
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_public_dict(self) -> dict:
        payload = {
            "ticker": self.ticker,
            "sentiment": round(self.sentiment, 2),
            "catalysts": self.catalysts,
            "risk_factors": self.risk_factors,
            "confidence": round(self.confidence, 2),
            "sources_used": self.sources_used,
            "as_of": self.as_of.isoformat(),
        }
        if self.scoring_snapshot is not None:
            payload["scoring_snapshot"] = self.scoring_snapshot
        return payload
