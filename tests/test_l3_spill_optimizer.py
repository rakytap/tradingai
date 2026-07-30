from datetime import datetime, timezone
from pathlib import Path

from tradingai.news.analysis.l3_spill_optimizer import (
    L3SpillOptimizationError,
    optimize_l3_spill_alpha,
)
from tradingai.news.analysis.l3_spillover import (
    apply_l3_spill_to_score,
    extract_l3_spill_inputs,
    rows_with_l3_spill_inputs,
)
from tradingai.news.config import (
    clear_l3_spill_cache,
    clear_news_scoring_cache,
    get_l3_news_spill_alpha,
    save_l3_spill_alpha,
)


def _row_with_spill(
    *,
    ticker: str,
    own: float,
    neighbor_score: float,
    actual_return_pct: float,
    technical: float = 0.0,
    regime: float = 0.0,
    ts: str,
) -> dict:
    return {
        "ticker": ticker,
        "timestamp": ts,
        "horizon_days": 5,
        "technical_score": technical,
        "regime_score": regime,
        "news_score": own,
        "actual_return_pct": actual_return_pct,
        "agent_payload": {
            "news": {
                "event_score": own,
                "l3_news_spill": {
                    "own": own,
                    "used": [
                        {"ticker": "SMH", "weight": 1.0, "score": neighbor_score},
                    ],
                    "adjusted": own,
                },
            }
        },
    }


def test_extract_and_apply_l3_spill_inputs():
    row = _row_with_spill(
        ticker="MSFT",
        own=0.0,
        neighbor_score=1.0,
        actual_return_pct=1.0,
        ts="2024-01-01T00:00:00+00:00",
    )
    inputs = extract_l3_spill_inputs(row)
    assert inputs is not None
    assert inputs["own"] == 0.0
    assert inputs["used"][0]["ticker"] == "SMH"
    # alpha=0.5 → 0.5*0 + 0.5*1 = 0.5
    assert apply_l3_spill_to_score(0.0, inputs, 0.5) == 0.5


def test_rows_with_l3_spill_inputs_filters():
    good = _row_with_spill(
        ticker="A",
        own=0.1,
        neighbor_score=0.2,
        actual_return_pct=1.0,
        ts="2024-01-01T00:00:00+00:00",
    )
    bad = {"ticker": "B", "agent_payload": {"news": {}}}
    assert rows_with_l3_spill_inputs([good, bad]) == [good]


def test_optimize_l3_spill_alpha_picks_better_alpha():
    # High alpha should help when neighbor score aligns with outcomes.
    rows = []
    for i in range(40):
        up = i % 2 == 0
        rows.append(
            _row_with_spill(
                ticker=f"T{i}",
                own=0.0,
                neighbor_score=1.0 if up else -1.0,
                actual_return_pct=2.0 if up else -2.0,
                technical=0.0,
                regime=0.0,
                ts=datetime(2024, 1, 1 + (i // 2), tzinfo=timezone.utc).isoformat(),
            )
        )
    result = optimize_l3_spill_alpha(
        rows,
        fusion_weights=(0.0, 1.0, 0.0),
        horizon_scales={5: 1.0},
        train_fraction=0.7,
        min_samples=30,
    )
    assert result["l3_news_spill_alpha"] >= 0.25
    assert result["tuned_validation"]["brier"] <= result["baseline_validation"]["brier"]


def test_optimize_l3_spill_requires_min_samples():
    rows = [
        _row_with_spill(
            ticker="A",
            own=0.0,
            neighbor_score=1.0,
            actual_return_pct=1.0,
            ts="2024-01-01T00:00:00+00:00",
        )
    ]
    try:
        optimize_l3_spill_alpha(
            rows,
            fusion_weights=(0.45, 0.35, 0.2),
            horizon_scales={5: 1.0},
            min_samples=30,
        )
        raised = False
    except L3SpillOptimizationError:
        raised = True
    assert raised


def test_get_l3_news_spill_alpha_prefers_json(tmp_path: Path, monkeypatch):
    path = tmp_path / "news_scoring.json"
    monkeypatch.setenv("TRADINGAI_NEWS_SCORING_ENABLED", "1")
    monkeypatch.setenv("TRADINGAI_NEWS_SCORING_PATH", str(path))
    monkeypatch.setenv("TRADINGAI_L3_NEWS_SPILL_ALPHA", "0.1")
    clear_news_scoring_cache()
    try:
        save_l3_spill_alpha(0.35, path=path)
        clear_l3_spill_cache()
        assert get_l3_news_spill_alpha() == 0.35
    finally:
        clear_news_scoring_cache()
