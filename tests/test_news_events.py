from tradingai.news.analysis.events import (
    NewsEvent,
    extract_news_events_from_text,
    propagate_themes_to_sectors,
    score_events_for_ticker,
)
from tradingai.predict.adapters.news import adapt_news_signal


def test_ai_capex_headline_maps_to_technology():
    events = extract_news_events_from_text(
        "Company boosts AI capex and GPU data center spend for next year"
    )
    assert len(events) == 1
    assert events[0].event_type == "ai_capex_increase"
    assert events[0].polarity == 1
    assert "ai_infra" in events[0].themes

    sectors = propagate_themes_to_sectors(events[0].themes)
    assert "Technology" in sectors
    assert sectors["Technology"] > 0

    scored = score_events_for_ticker(events, sector="Technology")
    assert scored["event_score"] is not None
    assert scored["event_score"] > 0
    assert "Technology" in scored["propagated_sectors"]


def test_export_restriction_negative_polarity_and_decay():
    events = extract_news_events_from_text(
        "US export ban and chip restriction hit semiconductor suppliers"
    )
    assert any(e.event_type == "export_restriction" for e in events)
    event = next(e for e in events if e.event_type == "export_restriction")
    assert event.polarity == -1

    # Direct geopolitics → Technology edge is negative; semiconductors → Technology positive.
    geo = propagate_themes_to_sectors(["geopolitics"])
    assert geo["Technology"] < 0

    semi = propagate_themes_to_sectors(["semiconductors"])
    assert semi["Technology"] == 1.0

    # Two-hop from ai_infra → semiconductors → Technology uses decay on second hop.
    two_hop = propagate_themes_to_sectors(["ai_infra"])
    # Direct: 0.8; via semiconductors: 0.9 * 1.0 * 0.5 = 0.45 → total Technology > 0.8
    assert two_hop["Technology"] >= 0.8 + 0.45 - 1e-9


def test_score_events_polarity_sign():
    beat = NewsEvent(
        event_type="earnings_beat",
        polarity=1,
        themes=["earnings_optimism"],
        confidence=0.8,
    )
    miss = NewsEvent(
        event_type="earnings_miss",
        polarity=-1,
        themes=["earnings_optimism"],
        confidence=0.8,
    )
    up = score_events_for_ticker([beat], sector="Technology")
    down = score_events_for_ticker([miss], sector="Technology")
    assert up["event_score"] > 0
    assert down["event_score"] < 0


def test_adapt_news_uses_event_score_over_sentiment():
    payload = {
        "ticker": "NVDA",
        "sentiment": 0.2,
        "confidence": 0.5,
        "event_score": 0.7,
        "event_confidence": 0.75,
        "event_reasons": ["ai_capex_increase (+1): ai_infra → Technology"],
        "events": [{"event_type": "ai_capex_increase", "polarity": 1}],
        "catalysts": ["ignored when events present"],
    }
    signal = adapt_news_signal("NVDA", payload)
    assert signal.directional_score == 0.7
    assert signal.raw["news_score_source"] == "events"
    assert signal.confidence == 0.75
    assert "ai_capex_increase" in signal.reasons[0]


def test_adapt_news_falls_back_to_sentiment_when_no_events():
    payload = {
        "ticker": "NVDA",
        "sentiment": 0.78,
        "confidence": 0.81,
        "catalysts": ["Positive earnings revisions"],
        "risk_factors": ["valuation"],
        "event_score": None,
        "events": [],
    }
    signal = adapt_news_signal("NVDA", payload)
    assert signal.directional_score == round((0.78 - 0.5) * 2.0, 4)
    assert signal.bullish_score == 0.78
    assert signal.raw["news_score_source"] == "sentiment"
    assert "Positive earnings revisions" in signal.reasons
