from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tradingai.predict.config import (
    clear_fusion_config_cache,
    get_fusion_weights,
    get_horizon_scales,
)
from tradingai.predict.fusion.optimizer import (
    FusionOptimizationError,
    brier_score,
    chronological_split,
    grid_weight_candidates,
    optimize_fusion,
    probabilities_for_rows,
)
from tradingai.predict.fusion.scorer import fuse_agent_signals
from tradingai.predict.models import AgentSignal


def _row(
    *,
    day: int,
    horizon: int,
    technical: float,
    news: float,
    regime: float,
    actual: float,
) -> dict:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "timestamp": timestamp,
        "horizon_days": horizon,
        "technical_score": technical,
        "news_score": news,
        "regime_score": regime,
        "actual_return_pct": actual,
    }


def _synthetic_dataset(*, rows_per_horizon: int = 40) -> list[dict]:
    rows: list[dict] = []
    for horizon in (3, 5, 8):
        for index in range(rows_per_horizon):
            technical = 0.6 if index % 2 == 0 else -0.4
            news = 0.3 if index % 3 == 0 else -0.2
            regime = 0.5
            actual = 1.5 if (technical + news + regime) > 0.4 else -1.0
            rows.append(
                _row(
                    day=index,
                    horizon=horizon,
                    technical=technical,
                    news=news,
                    regime=regime,
                    actual=actual,
                )
            )
    return rows


def test_chronological_split_preserves_order():
    rows = [
        _row(day=3, horizon=5, technical=0.1, news=0.1, regime=0.1, actual=1.0),
        _row(day=1, horizon=5, technical=0.2, news=0.2, regime=0.2, actual=1.0),
        _row(day=2, horizon=5, technical=0.3, news=0.3, regime=0.3, actual=1.0),
    ]
    train, validation = chronological_split(rows, train_fraction=0.67)
    assert [row["timestamp"].day for row in train] == [2, 3]
    assert [row["timestamp"].day for row in validation] == [4]


def test_grid_weight_candidates_respect_simplex():
    candidates = grid_weight_candidates(step=0.10)
    assert candidates
    for technical, news, regime in candidates:
        assert technical >= 0.10 - 1e-9
        assert news >= 0.10 - 1e-9
        assert regime >= 0.10 - 1e-9
        assert round(technical + news + regime, 4) == 1.0


def test_optimize_fusion_improves_or_matches_baseline():
    result = optimize_fusion(
        _synthetic_dataset(),
        horizons=(3, 5, 8),
        train_fraction=0.7,
        min_samples=30,
        weight_step=0.10,
    )
    assert "weights" in result
    assert "horizon_scales" in result
    assert result["tuned_validation"]["brier"] <= result["baseline_validation"]["brier"]


def test_optimize_fusion_requires_minimum_samples():
    rows = _synthetic_dataset(rows_per_horizon=5)
    with pytest.raises(FusionOptimizationError, match="need at least"):
        optimize_fusion(rows, horizons=(3, 5, 8), min_samples=30)


def test_brier_score_matches_manual_calculation():
    score = brier_score([0.8, 0.2], [1.0, -1.0])
    assert round(score, 4) == round(((0.8 - 1.0) ** 2 + (0.2 - 0.0) ** 2) / 2, 4)


def test_probabilities_for_rows_use_sigmoid():
    rows = [_row(day=1, horizon=5, technical=0.5, news=0.0, regime=0.0, actual=1.0)]
    probs = probabilities_for_rows(rows, (1.0, 0.0, 0.0), scale=1.0)
    assert len(probs) == 1
    assert 0.0 < probs[0] < 1.0


def test_config_loads_tuned_weights_and_scales(tmp_path, monkeypatch):
    tuning_path = tmp_path / "fusion_weights.json"
    tuning_path.write_text(
        json.dumps(
            {
                "weights": {"technical": 0.50, "news": 0.30, "regime": 0.20},
                "horizon_scales": {"3": 1.0, "5": 1.3, "8": 0.9},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADINGAI_FUSION_TUNING_PATH", str(tuning_path))
    monkeypatch.setenv("TRADINGAI_FUSION_TUNING_ENABLED", "1")
    monkeypatch.setenv("TRADINGAI_PREDICT_WEIGHTS", "0.10,0.10,0.80")
    clear_fusion_config_cache()

    assert get_fusion_weights() == (0.5, 0.3, 0.2)
    assert get_horizon_scales()[5] == 1.3
    assert get_horizon_scales()[8] == 0.9


def test_config_falls_back_to_env_when_tuning_disabled(tmp_path, monkeypatch):
    tuning_path = tmp_path / "fusion_weights.json"
    tuning_path.write_text(
        json.dumps({"weights": {"technical": 0.50, "news": 0.30, "regime": 0.20}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADINGAI_FUSION_TUNING_PATH", str(tuning_path))
    monkeypatch.setenv("TRADINGAI_FUSION_TUNING_ENABLED", "0")
    monkeypatch.setenv("TRADINGAI_PREDICT_WEIGHTS", "0.60,0.20,0.20")
    clear_fusion_config_cache()

    assert get_fusion_weights() == (0.6, 0.2, 0.2)


def test_scorer_uses_tuned_scales(tmp_path, monkeypatch):
    tuning_path = tmp_path / "fusion_weights.json"
    tuning_path.write_text(
        json.dumps(
            {
                "weights": {"technical": 0.45, "news": 0.35, "regime": 0.20},
                "horizon_scales": {"3": 2.0, "5": 2.0, "8": 2.0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADINGAI_FUSION_TUNING_PATH", str(tuning_path))
    monkeypatch.setenv("TRADINGAI_FUSION_TUNING_ENABLED", "1")
    clear_fusion_config_cache()

    signals = [
        AgentSignal(
            agent="technical",
            ticker="AAPL",
            directional_score=0.42,
            bullish_score=0.71,
            bearish_score=0.29,
            confidence=0.57,
            reasons=[],
            raw={},
        ),
        AgentSignal(
            agent="news",
            ticker="AAPL",
            directional_score=0.30,
            bullish_score=0.65,
            bearish_score=0.35,
            confidence=0.81,
            reasons=[],
            raw={},
        ),
        AgentSignal(
            agent="regime",
            ticker=None,
            directional_score=0.18,
            bullish_score=0.59,
            bearish_score=0.41,
            confidence=0.74,
            reasons=[],
            raw={},
        ),
    ]

    report = fuse_agent_signals("AAPL", signals)
    horizon_5 = next(item for item in report.horizons if item.horizon_days == 5)
    assert horizon_5.probability_up > 0.5
