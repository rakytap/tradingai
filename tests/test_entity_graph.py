from tradingai.news.analysis.events import NewsEvent
from tradingai.news.analysis.related_candidates import (
    resolve_related_candidates,
    ticker_graph_view,
)
from tradingai.news.config import clear_l3_spill_cache
from tradingai.news.entity_graph import get_node
from tradingai.predict.adapters.news import adapt_news_signal


def test_msft_ai_infra_expands_to_smh_and_power():
    events = [
        NewsEvent(
            event_type="ai_capex_increase",
            polarity=1,
            themes=["ai_infra"],
            confidence=0.8,
            subject_ticker="MSFT",
            evidence="MSFT boosts AI capex",
        )
    ]
    candidates = resolve_related_candidates("MSFT", events=events, limit=12)
    tickers = {c["ticker"] for c in candidates}
    assert "SMH" in tickers
    assert tickers & {"XLU", "VRT", "ETN", "CEG", "NEE"}
    assert "MSFT" not in tickers
    assert candidates[0]["weight"] >= candidates[-1]["weight"]


def test_missing_ticker_returns_empty():
    assert resolve_related_candidates("ZZZZ", themes=["ai_infra"]) == []


def test_dedupe_keeps_max_weight():
    events = [
        NewsEvent(
            event_type="ai_capex_increase",
            polarity=1,
            themes=["ai_infra", "semiconductors"],
            confidence=0.9,
        )
    ]
    candidates = resolve_related_candidates("MSFT", events=events, limit=20)
    by_ticker = {}
    for item in candidates:
        assert item["ticker"] not in by_ticker
        by_ticker[item["ticker"]] = item
    assert "NVDA" in by_ticker
    assert by_ticker["NVDA"]["relation"] == "supplier"
    assert by_ticker["NVDA"]["weight"] == 0.8


def test_ticker_graph_view():
    view = ticker_graph_view("MSFT")
    assert view["in_graph"] is True
    assert view["node"]["ticker"] == "MSFT"
    assert any(n["ticker"] == "NVDA" and n["relation"] == "supplier" for n in view["neighbors"])
    assert get_node("MSFT") is not None


def test_adapt_news_score_unchanged_when_spill_off(monkeypatch):
    monkeypatch.setenv("TRADINGAI_L3_NEWS_SPILL", "0")
    clear_l3_spill_cache()
    try:
        base = {
            "ticker": "MSFT",
            "sentiment": 0.5,
            "event_score": 0.55,
            "event_confidence": 0.7,
            "event_reasons": ["ai_capex"],
            "events": [{"event_type": "ai_capex_increase", "themes": ["ai_infra"]}],
        }
        without = adapt_news_signal("MSFT", base)
        with_cands = adapt_news_signal(
            "MSFT",
            {
                **base,
                "related_candidates": [
                    {"ticker": "SMH", "relation": "thematic_proxy", "weight": 0.9, "reason": "x"}
                ],
            },
        )
        assert without.directional_score == with_cands.directional_score == 0.55
        assert with_cands.raw["related_candidates"][0]["ticker"] == "SMH"
        assert "l3_news_spill" not in with_cands.raw
    finally:
        clear_l3_spill_cache()


def test_adapt_news_spill_moves_toward_neighbor(monkeypatch):
    monkeypatch.setenv("TRADINGAI_L3_NEWS_SPILL", "1")
    monkeypatch.setenv("TRADINGAI_L3_NEWS_SPILL_ALPHA", "0.25")
    clear_l3_spill_cache()

    def fake_load(candidates, **_kwargs):
        return {"SMH": 1.0}

    monkeypatch.setattr(
        "tradingai.news.analysis.l3_spillover.load_neighbor_news_scores",
        fake_load,
    )
    try:
        signal = adapt_news_signal(
            "MSFT",
            {
                "ticker": "MSFT",
                "sentiment": 0.5,
                "event_score": 0.0,
                "event_confidence": 0.7,
                "event_reasons": ["ai_capex"],
                "events": [{"event_type": "ai_capex_increase", "themes": ["ai_infra"]}],
                "related_candidates": [
                    {"ticker": "SMH", "relation": "thematic_proxy", "weight": 0.9, "reason": "x"}
                ],
            },
        )
        # (1-0.25)*0 + 0.25*1.0 = 0.25
        assert signal.directional_score == 0.25
        assert signal.raw["event_score"] == 0.0
        assert signal.raw["l3_news_spill"]["adjusted"] == 0.25
        assert signal.raw["l3_news_spill"]["used"][0]["ticker"] == "SMH"
    finally:
        clear_l3_spill_cache()
