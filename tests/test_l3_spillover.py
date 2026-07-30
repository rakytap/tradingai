from tradingai.news.analysis.l3_spillover import (
    blend_l3_news_score,
    neighbor_directional_from_snapshot,
)


def test_blend_weights_and_alpha():
    candidates = [
        {"ticker": "SMH", "weight": 0.9},
        {"ticker": "NVDA", "weight": 0.1},
    ]
    neighbor_scores = {"SMH": 1.0, "NVDA": -1.0}
    adjusted, meta = blend_l3_news_score(0.0, candidates, neighbor_scores, alpha=0.25)
    # neighbor_avg = 0.9*1 + 0.1*(-1) = 0.8
    # adjusted = 0.75*0 + 0.25*0.8 = 0.2
    assert abs(adjusted - 0.2) < 1e-9
    assert meta["neighbor_avg"] == 0.8
    assert len(meta["used"]) == 2


def test_blend_ignores_missing_neighbors():
    candidates = [
        {"ticker": "SMH", "weight": 0.9},
        {"ticker": "ZZZZ", "weight": 0.9},
    ]
    adjusted, meta = blend_l3_news_score(
        0.4, candidates, {"SMH": 1.0}, alpha=0.5
    )
    # only SMH: 0.5*0.4 + 0.5*1.0 = 0.7
    assert abs(adjusted - 0.7) < 1e-9
    assert [u["ticker"] for u in meta["used"]] == ["SMH"]


def test_blend_alpha_zero_unchanged():
    adjusted, meta = blend_l3_news_score(
        0.55,
        [{"ticker": "SMH", "weight": 0.9}],
        {"SMH": 1.0},
        alpha=0.0,
    )
    assert adjusted == 0.55
    assert meta["adjusted"] == 0.55


def test_blend_clips_to_unit_interval():
    adjusted, _ = blend_l3_news_score(
        1.0,
        [{"ticker": "SMH", "weight": 1.0}],
        {"SMH": 1.0},
        alpha=1.0,
    )
    assert adjusted == 1.0


def test_neighbor_directional_prefers_event_score():
    assert neighbor_directional_from_snapshot({"event_score": 0.6, "sentiment": 0.1}) == 0.6


def test_neighbor_directional_falls_back_to_sentiment():
    assert neighbor_directional_from_snapshot({"sentiment": 0.75}) == 0.5
