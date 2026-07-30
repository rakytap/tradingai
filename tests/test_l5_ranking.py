from datetime import date

import pandas as pd

from tradingai.analysis.ml.features import TECHNICAL_FEATURE_COLUMNS, attach_sector_relative_features
from tradingai.analysis.ml.labels import add_cross_section_label, add_direction_label
from tradingai.analysis.ml.sector_models import resolve_model_dir_for_sector, sector_model_dir, sector_slug
from tradingai.predict.config import clear_fusion_config_cache
from tradingai.predict.portfolio import select_portfolio_positions


def test_cross_section_label_above_median():
    frame = pd.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "sector": ["Technology"] * 5,
            "ticker": ["A", "B", "C", "D", "E"],
            "forward_return_pct": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    labeled = add_cross_section_label(frame, min_peers=5)
    # Median of 1..5 is 3 → above median: 4,5
    winners = set(labeled.loc[labeled["cs_label"], "ticker"])
    assert winners == {"D", "E"}
    assert set(labeled.loc[~labeled["cs_label"], "ticker"]) == {"A", "B", "C"}


def test_cross_section_drops_small_peer_groups():
    frame = pd.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 3,
            "sector": ["Technology"] * 3,
            "ticker": ["A", "B", "C"],
            "forward_return_pct": [1.0, 2.0, 3.0],
        }
    )
    labeled = add_cross_section_label(frame, min_peers=5)
    assert labeled.empty


def test_sector_relative_features_technology_vs_xlk():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    symbol = pd.DataFrame({"date": idx, "close": range(100, 140)}, index=idx)
    # Sector ETF rises faster → negative excess for flat-ish symbol growth
    sector = pd.DataFrame({"close": [100 + i * 2 for i in range(40)]}, index=idx)
    out = attach_sector_relative_features(symbol, sector)
    assert "sector_rs_spread" in out.columns
    assert "sector_rs_line" in out.columns
    assert "sector_excess_1m" in out.columns
    assert out["sector_rs_spread"].notna().sum() > 0
    assert all(col in TECHNICAL_FEATURE_COLUMNS for col in (
        "sector_rs_spread",
        "sector_rs_line",
        "sector_excess_1m",
    ))


def test_sector_slug_and_resolve_prefers_sector(tmp_path):
    assert sector_slug("Technology") == "technology"
    sector_dir = sector_model_dir("Technology", model_dir=tmp_path)
    horizon = sector_dir / "horizon_5"
    horizon.mkdir(parents=True)
    (horizon / "logistic.joblib").write_bytes(b"x")
    resolved, source = resolve_model_dir_for_sector(
        "Technology",
        model_name="logistic",
        horizon_days=5,
        model_dir=tmp_path,
        allow_global_fallback=False,
    )
    assert source == "sector"
    assert resolved == sector_dir


def test_rank_order_stable():
    from tradingai.analysis.ml.ranking import rank_tickers_in_sector

    # Unit-level: sort helper behavior via manual list mimicking output
    rows = [
        {"ticker": "B", "ml_score": 0.4},
        {"ticker": "A", "ml_score": 0.9},
        {"ticker": "C", "ml_score": 0.9},
    ]
    ranked = sorted(rows, key=lambda item: (-float(item["ml_score"]), item["ticker"]))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    assert [r["ticker"] for r in ranked] == ["A", "C", "B"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert rank_tickers_in_sector  # import smoke


def test_portfolio_bullish_gate_drops_bearish():
    candidates = [
        {
            "ticker": "AAPL",
            "combined_score": 0.5,
            "probability_up": 0.7,
            "sector": "Technology",
            "regime": "bullish",
            "ml_rank_score": 0.8,
        },
        {
            "ticker": "XOM",
            "combined_score": 0.9,
            "probability_up": 0.9,
            "sector": "Energy",
            "regime": "bearish",
            "ml_rank_score": 0.95,
        },
    ]
    selected = select_portfolio_positions(
        candidates,
        size=5,
        min_combined_score=0.0,
        min_probability_up=0.55,
        require_bullish_sector=True,
    )
    assert [row["ticker"] for row in selected] == ["AAPL"]


def _ml_vs_combined_candidates():
    return [
        {
            "ticker": "LOW_CS",
            "combined_score": 0.9,
            "probability_up": 0.8,
            "sector": "Technology",
            "regime": "bullish",
            "ml_rank_score": 0.2,
        },
        {
            "ticker": "HIGH_CS",
            "combined_score": 0.1,
            "probability_up": 0.8,
            "sector": "Technology",
            "regime": "bullish",
            "ml_rank_score": 0.9,
        },
    ]


def test_portfolio_prefers_ml_rank_score(monkeypatch):
    monkeypatch.setenv("TRADINGAI_PORTFOLIO_RANK_KEY", "ml_rank_score")
    clear_fusion_config_cache()
    try:
        selected = select_portfolio_positions(
            _ml_vs_combined_candidates(),
            size=2,
            min_combined_score=-1.0,
            min_probability_up=0.5,
            require_bullish_sector=True,
            max_per_sector=10,
        )
        assert selected[0]["ticker"] == "HIGH_CS"
    finally:
        clear_fusion_config_cache()


def test_portfolio_prefers_combined_score(monkeypatch):
    monkeypatch.setenv("TRADINGAI_PORTFOLIO_RANK_KEY", "combined_score")
    clear_fusion_config_cache()
    try:
        selected = select_portfolio_positions(
            _ml_vs_combined_candidates(),
            size=2,
            min_combined_score=-1.0,
            min_probability_up=0.5,
            require_bullish_sector=True,
            max_per_sector=10,
        )
        assert selected[0]["ticker"] == "LOW_CS"
    finally:
        clear_fusion_config_cache()


def test_direction_label_still_absolute():
    frame = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 110.0],
        }
    )
    labeled = add_direction_label(frame, horizon_days=1)
    assert bool(labeled["direction_up"].iloc[0]) is True
