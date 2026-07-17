from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tradingai.news.analysis.scoring_optimizer import (
    NewsScoringOptimizationError,
    build_scoring_params,
    extract_scoring_snapshot,
    optimize_news_scoring,
    resimulate_news_scores,
    sentiment_to_directional,
)
from tradingai.news.analysis.sentiment_scorer import recency_weight, rescore_from_snapshot
from tradingai.news.analysis.synthesizer import synthesize_sentiment_report
from tradingai.news.config import (
    clear_news_scoring_cache,
    get_news_scoring_params,
)
from tradingai.news.models import AnalystAction, NewsItem


FIXED_NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def test_recency_weight_exponential_and_floor():
    recent = FIXED_NOW - timedelta(days=1)
    old = FIXED_NOW - timedelta(days=60)
    params = build_scoring_params(
        source_weights={"news": 1.0},
        half_life_days=7.0,
        floor=0.4,
    )
    recent_w = recency_weight(recent, FIXED_NOW, params=params)
    old_w = recency_weight(old, FIXED_NOW, params=params)
    missing_w = recency_weight(None, FIXED_NOW, params=params)
    assert recent_w > old_w
    assert old_w == pytest.approx(0.4)
    assert missing_w == pytest.approx(0.6)


def test_rescore_from_snapshot_responds_to_half_life_and_weights():
    snapshot = {
        "items": [
            {
                "channel": "news",
                "base_score": 0.9,
                "published_at": (FIXED_NOW - timedelta(days=1)).isoformat(),
                "engagement": 1.0,
            },
            {
                "channel": "social",
                "base_score": 0.1,
                "published_at": (FIXED_NOW - timedelta(days=1)).isoformat(),
                "engagement": 1.0,
            },
        ]
    }
    high_social = build_scoring_params(
        source_weights={
            "news": 1.0,
            "press": 1.1,
            "analyst": 1.4,
            "filings": 0.9,
            "earnings": 1.3,
            "transcript": 1.0,
            "social": 2.0,
        },
        half_life_days=7.0,
    )
    low_social = build_scoring_params(
        source_weights={
            "news": 1.0,
            "press": 1.1,
            "analyst": 1.4,
            "filings": 0.9,
            "earnings": 1.3,
            "transcript": 1.0,
            "social": 0.1,
        },
        half_life_days=7.0,
    )
    high = rescore_from_snapshot(snapshot, high_social, as_of=FIXED_NOW)
    low = rescore_from_snapshot(snapshot, low_social, as_of=FIXED_NOW)
    assert low > high


def test_synthesize_attaches_scoring_snapshot():
    report = synthesize_sentiment_report(
        ticker="NVDA",
        news=[
            NewsItem(
                title="Strong product launch drives demand",
                summary="Growth accelerated.",
                source="Test",
                provider="yfinance",
                published_at=FIXED_NOW - timedelta(days=1),
            )
        ],
        press_releases=[],
        filings=[],
        analyst_actions=[
            AnalystAction(
                firm="Example",
                action="upgrade",
                provider="yfinance",
                published_at=FIXED_NOW - timedelta(days=1),
            )
        ],
        sources_used=["yfinance"],
    )
    payload = report.to_public_dict()
    assert "scoring_snapshot" in payload
    assert payload["scoring_snapshot"]["items"]
    channels = {item["channel"] for item in payload["scoring_snapshot"]["items"]}
    assert "news" in channels
    assert "analyst" in channels


def test_config_loads_news_scoring_file(tmp_path, monkeypatch):
    path = tmp_path / "news_scoring.json"
    path.write_text(
        json.dumps(
            {
                "source_weights": {
                    "news": 1.2,
                    "press": 1.0,
                    "analyst": 1.5,
                    "filings": 0.8,
                    "earnings": 1.4,
                    "transcript": 1.0,
                    "social": 0.3,
                },
                "recency": {"half_life_days": 10.0, "floor": 0.4, "missing_date_weight": 0.6},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADINGAI_NEWS_SCORING_PATH", str(path))
    monkeypatch.setenv("TRADINGAI_NEWS_SCORING_ENABLED", "1")
    clear_news_scoring_cache()
    params = get_news_scoring_params()
    assert params["source_weights"]["social"] == 0.3
    assert params["recency"]["half_life_days"] == 10.0


def _synthetic_snapshot_rows(*, count: int = 40) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        bullish = index % 2 == 0
        base_news = 0.85 if bullish else 0.2
        base_social = 0.7 if bullish else 0.3
        actual = 1.5 if bullish else -1.2
        published = (FIXED_NOW - timedelta(days=count - index)).isoformat()
        snapshot = {
            "items": [
                {
                    "channel": "news",
                    "base_score": base_news,
                    "published_at": published,
                    "engagement": 1.0,
                },
                {
                    "channel": "social",
                    "base_score": base_social,
                    "published_at": published,
                    "engagement": 1.0,
                },
            ]
        }
        rows.append(
            {
                "timestamp": FIXED_NOW - timedelta(days=count - index),
                "horizon_days": 5,
                "technical_score": 0.2 if bullish else -0.2,
                "news_score": sentiment_to_directional(0.5),
                "regime_score": 0.1,
                "actual_return_pct": actual,
                "agent_payload": {"news": {"scoring_snapshot": snapshot}},
            }
        )
    return rows


def test_optimize_news_scoring_improves_or_matches_baseline():
    result = optimize_news_scoring(
        _synthetic_snapshot_rows(),
        fusion_weights=(0.45, 0.35, 0.20),
        horizon_scales={5: 1.2},
        train_fraction=0.7,
        min_samples=30,
    )
    assert "source_weights" in result
    assert "recency" in result
    assert result["tuned_validation"]["brier"] <= result["baseline_validation"]["brier"]


def test_optimize_news_scoring_requires_snapshots():
    rows = [
        {
            "timestamp": FIXED_NOW,
            "horizon_days": 5,
            "technical_score": 0.1,
            "news_score": 0.1,
            "regime_score": 0.1,
            "actual_return_pct": 1.0,
            "agent_payload": {"news": {}},
        }
    ]
    with pytest.raises(NewsScoringOptimizationError, match="scoring_snapshot"):
        optimize_news_scoring(
            rows,
            fusion_weights=(0.45, 0.35, 0.20),
            horizon_scales={5: 1.2},
            min_samples=30,
        )


def test_extract_and_resimulate_news_scores():
    rows = _synthetic_snapshot_rows(count=5)
    assert extract_scoring_snapshot(rows[0]) is not None
    params = build_scoring_params(
        source_weights={
            "news": 1.0,
            "press": 1.1,
            "analyst": 1.4,
            "filings": 0.9,
            "earnings": 1.3,
            "transcript": 1.0,
            "social": 0.45,
        },
        half_life_days=7.0,
    )
    updated = resimulate_news_scores(rows, params)
    assert updated[0]["news_score"] != 0.0 or updated[1]["news_score"] != 0.0
