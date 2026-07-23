import pytest

from tradingai.predict.portfolio import select_portfolio_positions


def _row(
    ticker: str,
    *,
    combined: float,
    prob: float,
    sector: str = "Technology",
) -> dict:
    return {
        "ticker": ticker,
        "combined_score": combined,
        "probability_up": prob,
        "sector": sector,
    }


def test_select_portfolio_limits_sector_and_broad_etf():
    candidates = [
        _row("SPY", combined=0.9, prob=0.7, sector="Broad"),
        _row("VOO", combined=0.85, prob=0.68, sector="Broad"),
        _row("AAPL", combined=0.8, prob=0.65),
        _row("MSFT", combined=0.75, prob=0.62),
        _row("NVDA", combined=0.7, prob=0.6),
        _row("AMD", combined=0.65, prob=0.58),
    ]
    selected = select_portfolio_positions(
        candidates,
        size=4,
        min_probability_up=0.55,
        max_per_sector=2,
        max_broad_etf=1,
    )
    tickers = [row["ticker"] for row in selected]
    assert "SPY" in tickers or "VOO" in tickers
    assert not ("SPY" in tickers and "VOO" in tickers)
    assert len(selected) == 3
    assert tickers == ["AAPL", "MSFT", "SPY"]


def test_build_paper_portfolio_requires_predictions(monkeypatch):
    from tradingai.predict.portfolio import build_paper_portfolio

    monkeypatch.setattr(
        "tradingai.predict.portfolio._latest_predictions_per_ticker",
        lambda **kwargs: [],
    )
    with pytest.raises(ValueError, match="No stored predictions"):
        build_paper_portfolio(horizon_days=99)
