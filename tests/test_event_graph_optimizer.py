from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from tradingai.news.analysis.event_graph_optimizer import (
    EventGraphOptimizationError,
    event_directional_for_row,
    optimize_event_graph,
    resimulate_fused_news_scores,
    resolve_news_score_for_row,
)
from tradingai.news.analysis.events import (
    default_event_graph_params,
    extract_news_events_from_text,
    score_events_for_ticker,
)
from tradingai.news.analysis.synthesizer import synthesize_sentiment_report
from tradingai.news.config import clear_event_graph_cache, get_event_graph_params
from tradingai.news.models import NewsItem


FIXED_NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _graph_with_decay(decay: float) -> dict:
    params = default_event_graph_params()
    params["propagation_decay"] = decay
    return params


def test_event_score_responds_to_decay_and_divisor():
    events = extract_news_events_from_text(
        "Company boosts AI capex and GPU data center spend for next year"
    )
    low = score_events_for_ticker(
        deepcopy(events),
        sector="Technology",
        graph_params=_graph_with_decay(0.3),
    )
    high = score_events_for_ticker(
        deepcopy(events),
        sector="Technology",
        graph_params=_graph_with_decay(0.7),
    )
    assert low["event_score"] is not None
    assert high["event_score"] is not None

    tight = default_event_graph_params()
    tight["score_divisor"] = 1.0
    loose = default_event_graph_params()
    loose["score_divisor"] = 2.0
    tight_score = score_events_for_ticker(
        deepcopy(events), sector="Technology", graph_params=tight
    )["event_score"]
    loose_score = score_events_for_ticker(
        deepcopy(events), sector="Technology", graph_params=loose
    )["event_score"]
    assert abs(tight_score) >= abs(loose_score)


def test_disable_event_type_zeros_score_from_payload():
    params = default_event_graph_params()
    for spec in params["event_specs"]:
        if spec["event_type"] == "earnings_beat":
            spec["enabled"] = False

    row = {
        "ticker": "AAPL",
        "sector": "Technology",
        "agent_payload": {
            "news": {
                "events": [
                    {
                        "event_type": "earnings_beat",
                        "polarity": 1,
                        "themes": ["earnings_optimism"],
                        "confidence": 0.8,
                    }
                ]
            }
        },
    }
    assert event_directional_for_row(row, params) is None


def test_pattern_ablation_changes_extraction():
    params = default_event_graph_params()
    text = "Company beat estimates and raised guidance"
    before = extract_news_events_from_text(text, graph_params=params)
    assert any(e.event_type == "earnings_beat" for e in before)

    for spec in params["event_specs"]:
        if spec["event_type"] == "earnings_beat":
            spec["patterns"] = [r"this pattern will never match xyz"]

    after = extract_news_events_from_text(text, graph_params=params)
    assert not any(e.event_type == "earnings_beat" for e in after)


def test_synthesize_attaches_event_text_corpus():
    report = synthesize_sentiment_report(
        "NVDA",
        news=[
            NewsItem(
                title="Company boosts AI capex and GPU data center spend",
                summary="Expansion planned",
                source="TestWire",
                published_at=FIXED_NOW,
                provider="yfinance",
            )
        ],
        press_releases=[],
        filings=[],
        analyst_actions=[],
        sources_used=["yfinance"],
    )
    assert report.event_text_corpus
    assert "AI capex" in report.event_text_corpus[0]["text"]
    payload = report.to_public_dict()
    assert "event_text_corpus" in payload


def _synthetic_event_rows(count: int = 40) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        bullish = index % 2 == 0
        text = (
            "Company beat estimates and raised guidance"
            if bullish
            else "Company missed expectations and cut guidance"
        )
        actual = 1.5 if bullish else -1.2
        rows.append(
            {
                "ticker": "MSFT",
                "sector": "Technology",
                "timestamp": FIXED_NOW - timedelta(days=count - index),
                "horizon_days": 5,
                "technical_score": 0.2 if bullish else -0.2,
                "news_score": 0.0,
                "regime_score": 0.1,
                "actual_return_pct": actual,
                "agent_payload": {
                    "news": {
                        "event_text_corpus": [{"text": text, "source": "test"}],
                        "events": [],
                        "scoring_snapshot": {
                            "items": [
                                {
                                    "channel": "news",
                                    "base_score": 0.8 if bullish else 0.2,
                                    "published_at": (
                                        FIXED_NOW - timedelta(days=1)
                                    ).isoformat(),
                                    "engagement": 1.0,
                                }
                            ]
                        },
                    }
                },
            }
        )
    return rows


def test_optimize_event_graph_improves_or_matches_baseline():
    result = optimize_event_graph(
        _synthetic_event_rows(),
        fusion_weights=(0.45, 0.35, 0.20),
        horizon_scales={5: 1.2},
        train_fraction=0.7,
        min_samples=30,
    )
    assert "theme_edges" in result
    assert "event_specs" in result
    assert result["tuned_validation"]["brier"] <= result["baseline_validation"]["brier"]


def test_optimize_event_graph_requires_events():
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
    with pytest.raises(EventGraphOptimizationError, match="events"):
        optimize_event_graph(
            rows,
            fusion_weights=(0.45, 0.35, 0.20),
            horizon_scales={5: 1.2},
            min_samples=30,
        )


def test_resimulate_prefers_events_over_sentiment():
    rows = _synthetic_event_rows(count=2)
    params = default_event_graph_params()
    news_params = {
        "source_weights": {
            "news": 1.0,
            "press": 1.1,
            "analyst": 1.4,
            "filings": 0.9,
            "earnings": 1.3,
            "transcript": 1.0,
            "social": 0.45,
        },
        "recency": {
            "half_life_days": 7.0,
            "floor": 0.4,
            "missing_date_weight": 0.6,
        },
    }
    event_score = resolve_news_score_for_row(
        rows[0],
        news_params=news_params,
        event_graph_params=params,
    )
    sentiment_only = resolve_news_score_for_row(
        rows[0],
        news_params=news_params,
        event_graph_params=None,
    )
    assert event_score is not None
    assert sentiment_only is not None
    # Bullish earnings_beat should be positive via events.
    assert event_score > 0

    updated = resimulate_fused_news_scores(
        rows,
        news_params=news_params,
        event_graph_params=params,
    )
    assert updated[0]["news_score"] == event_score


def test_get_event_graph_params_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAI_EVENT_GRAPH_PATH", str(tmp_path / "missing.json"))
    clear_event_graph_cache()
    params = get_event_graph_params()
    assert params["propagation_decay"] == 0.5
    assert any(spec["event_type"] == "earnings_beat" for spec in params["event_specs"])
    clear_event_graph_cache()
