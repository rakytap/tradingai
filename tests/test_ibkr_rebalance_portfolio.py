from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingai.predict.portfolio_risk import (
    optimize_markowitz_rebalance_weights,
    optimize_markowitz_weights,
)


def test_rebalance_high_gamma_stays_near_prior():
    mu = np.array([0.01, 0.0001, 0.0001])
    sigma = np.diag([0.0004, 0.0004, 0.0004])
    prior = np.array([0.1, 0.4, 0.4])
    prices = np.array([50.0, 50.0, 50.0])

    greenfield = optimize_markowitz_weights(
        mu,
        sigma,
        invested_fraction=0.9,
        risk_aversion=1.0,
        max_weight=0.8,
    )
    sticky = optimize_markowitz_rebalance_weights(
        mu,
        sigma,
        prior,
        prices,
        invested_fraction=0.9,
        notional_usd=100_000.0,
        commission_preference=500.0,
        rate_per_share=0.0035,
        risk_aversion=1.0,
        max_weight=0.8,
    )

    assert float(np.sum(sticky)) == pytest.approx(0.9, abs=1e-5)
    assert float(np.linalg.norm(sticky - prior)) < float(np.linalg.norm(greenfield - prior))


def test_rebalance_small_book_suppresses_sub_minimum_tickets():
    """On a ~$600 book, $0.35 mins make micro-trades uneconomic at gamma=1."""
    mu = np.array([0.002, 0.0015, 0.0012])
    sigma = np.diag([0.0004, 0.0005, 0.00045])
    prior = np.array([0.4, 0.35, 0.25])
    prices = np.array([100.0, 50.0, 80.0])

    sticky = optimize_markowitz_rebalance_weights(
        mu,
        sigma,
        prior,
        prices,
        invested_fraction=1.0,
        notional_usd=615.0,
        commission_preference=1.0,
        rate_per_share=0.0035,
        risk_aversion=3.0,
        max_weight=0.5,
    )
    trade_usd = np.abs(sticky - prior) * 615.0
    material = trade_usd[trade_usd > 1e-9]
    # Sub-$70 tickets are snapped to prior (IBKR min never fully binds below that).
    assert material.size == 0 or float(np.min(material)) >= 70.0 - 1e-6
    assert float(np.sum(sticky)) == pytest.approx(1.0, abs=1e-5)


def test_build_ibkr_rebalance_requires_prior(tmp_path: Path):
    from tradingai.predict.portfolio import build_paper_portfolio_ibkr_rebalance

    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        build_paper_portfolio_ibkr_rebalance(previous_portfolio_path=missing)


def test_build_paper_portfolio_ibkr_rebalance(monkeypatch, tmp_path: Path):
    from tradingai.predict import portfolio as portfolio_module

    prior = {
        "holdings": [
            {
                "ticker": "AAA",
                "weight": 0.45,
                "probability_up": 0.6,
                "expected_return_pct": 3.0,
                "confidence": 0.5,
                "combined_score": 0.4,
                "technical_score": 0.2,
                "news_score": 0.2,
                "regime_score": 0.2,
                "sector": "Technology",
            },
            {
                "ticker": "BBB",
                "weight": 0.45,
                "probability_up": 0.55,
                "expected_return_pct": 1.0,
                "confidence": 0.5,
                "combined_score": 0.3,
                "technical_score": 0.2,
                "news_score": 0.2,
                "regime_score": 0.2,
                "sector": "Healthcare",
            },
        ]
    }
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior))

    selected = [
        {
            "id": 1,
            "ticker": "AAA",
            "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
            "horizon_days": 5,
            "probability_up": 0.6,
            "expected_return_pct": 4.0,
            "confidence": 0.7,
            "combined_score": 0.5,
            "sector": "Technology",
            "regime": "bull",
            "prediction_type": "momentum",
            "technical_score": 0.4,
            "news_score": 0.3,
            "regime_score": 0.2,
            "model_name": None,
        },
        {
            "id": 2,
            "ticker": "CCC",
            "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
            "horizon_days": 5,
            "probability_up": 0.62,
            "expected_return_pct": 5.0,
            "confidence": 0.7,
            "combined_score": 0.55,
            "sector": "Energy",
            "regime": "bull",
            "prediction_type": "momentum",
            "technical_score": 0.4,
            "news_score": 0.3,
            "regime_score": 0.2,
            "model_name": None,
        },
    ]

    index = pd.date_range("2024-01-01", periods=90, freq="B")
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.001, 0.01, size=len(index)),
            "BBB": rng.normal(0.0005, 0.02, size=len(index)),
            "CCC": rng.normal(0.0012, 0.015, size=len(index)),
        },
        index=index,
    )
    prices = {"AAA": 100.0, "BBB": 50.0, "CCC": 80.0}

    monkeypatch.setattr(
        portfolio_module,
        "_latest_predictions_per_ticker",
        lambda **kwargs: selected,
    )
    monkeypatch.setattr(
        portfolio_module,
        "select_portfolio_positions",
        lambda candidates, **kwargs: candidates,
    )
    monkeypatch.setattr(
        portfolio_module,
        "build_returns_frame",
        lambda tickers, **kwargs: returns[[t for t in tickers if t in returns.columns]],
    )
    monkeypatch.setattr(
        portfolio_module,
        "_latest_close",
        lambda ticker, **kwargs: prices.get(str(ticker).upper()),
    )

    result = portfolio_module.build_paper_portfolio_ibkr_rebalance(
        previous_portfolio_path=prior_path,
        commission_preference=1.0,
        horizon_days=5,
        positions=3,
        cash_weight=0.1,
        notional_usd=100_000.0,
    )
    assert result["optimizer"] == "markowitz_ibkr_rebalance"
    assert result["optimizer_status"] in {"ok", "fallback_markowitz"}
    assert "estimated_ibkr_commission_usd" in result
    assert result["commission_preference"] == 1.0
    tickers = {h["ticker"] for h in result["holdings"]}
    assert "BBB" in tickers  # prior-only name retained in universe
    weight_sum = sum(h["weight"] for h in result["holdings"])
    assert weight_sum == pytest.approx(0.9, abs=1e-3)
    for holding in result["holdings"]:
        assert "prior_weight" in holding
        assert "trade_usd" in holding
        assert "est_commission_usd" in holding


def test_mcp_ibkr_rebalance_tool_delegates(monkeypatch, tmp_path: Path):
    from tradingai.predict.tools import predict as predict_tools

    sentinel = {"optimizer": "markowitz_ibkr_rebalance", "ok": True}

    def _fake(**kwargs):
        assert kwargs["previous_portfolio_path"].endswith("prior.json")
        assert kwargs["commission_preference"] == 2.5
        return sentinel

    monkeypatch.setattr(predict_tools, "build_paper_portfolio_ibkr_rebalance", _fake)
    out = predict_tools.get_paper_portfolio_ibkr_rebalance_tool(
        previous_portfolio_path=str(tmp_path / "prior.json"),
        commission_preference=2.5,
    )
    assert out is sentinel
