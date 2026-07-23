from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tradingai.predict.portfolio_risk import (
    annualized_volatility_pct,
    estimate_mu_sigma,
    horizon_return_to_daily_mu,
    optimize_markowitz_weights,
)


def test_horizon_return_to_daily_mu_roundtrip():
    daily = horizon_return_to_daily_mu(10.0, horizon_days=5)
    compounded = (1.0 + daily) ** 5 - 1.0
    assert compounded == pytest.approx(0.10, rel=1e-9)


def test_optimize_markowitz_long_only_and_sum():
    mu = np.array([0.002, 0.001, 0.0005])
    # Diagonal covariance: asset 0 lowest vol, highest mu.
    sigma = np.diag([0.0001, 0.0004, 0.0009])
    weights = optimize_markowitz_weights(
        mu,
        sigma,
        invested_fraction=0.9,
        risk_aversion=3.0,
        max_weight=0.5,
    )
    assert weights.shape == (3,)
    assert np.all(weights >= -1e-10)
    assert float(weights.sum()) == pytest.approx(0.9, abs=1e-6)
    assert float(weights[0]) > float(weights[2])
    assert float(np.max(weights)) <= 0.5 + 1e-8


def test_optimize_markowitz_prefers_lower_vol_when_mu_equal():
    mu = np.array([0.001, 0.001])
    sigma = np.diag([0.0001, 0.001])
    weights = optimize_markowitz_weights(
        mu,
        sigma,
        invested_fraction=1.0,
        risk_aversion=5.0,
        max_weight=1.0,
    )
    assert float(weights[0]) > float(weights[1])


def test_estimate_mu_sigma_and_vol_display():
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.001, 0.01, size=len(index)),
            "BBB": rng.normal(0.0005, 0.02, size=len(index)),
        },
        index=index,
    )
    rows = [
        {
            "ticker": "AAA",
            "expected_return_pct": 5.0,
            "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
        },
        {
            "ticker": "BBB",
            "expected_return_pct": 2.0,
            "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
        },
    ]
    mu, sigma, tickers = estimate_mu_sigma(returns, rows, horizon_days=5)
    assert tickers == ["AAA", "BBB"]
    assert mu.shape == (2,)
    assert sigma.shape == (2, 2)
    vol_a = annualized_volatility_pct(sigma, 0)
    vol_b = annualized_volatility_pct(sigma, 1)
    assert vol_b > vol_a


def test_build_paper_portfolio_markowitz_branch(monkeypatch):
    from tradingai.predict import portfolio as portfolio_module

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
            "ticker": "BBB",
            "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
            "horizon_days": 5,
            "probability_up": 0.58,
            "expected_return_pct": 2.0,
            "confidence": 0.65,
            "combined_score": 0.35,
            "sector": "Healthcare",
            "regime": "bull",
            "prediction_type": "momentum",
            "technical_score": 0.2,
            "news_score": 0.2,
            "regime_score": 0.2,
            "model_name": None,
        },
    ]

    index = pd.date_range("2024-01-01", periods=90, freq="B")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.001, 0.01, size=len(index)),
            "BBB": rng.normal(0.0005, 0.03, size=len(index)),
        },
        index=index,
    )

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
        lambda tickers, **kwargs: returns,
    )

    result = portfolio_module.build_paper_portfolio(
        horizon_days=5,
        positions=2,
        cash_weight=0.1,
        optimizer="markowitz",
        notional_usd=100_000.0,
    )
    assert result["optimizer"] == "markowitz"
    assert result["optimizer_status"] == "ok"
    assert result["lookback_days"] == 126
    assert len(result["holdings"]) == 2
    weight_sum = sum(holding["weight"] for holding in result["holdings"])
    assert weight_sum == pytest.approx(0.9, abs=1e-3)
    for holding in result["holdings"]:
        assert holding["optimizer"] == "markowitz"
        assert "annualized_volatility_pct" in holding
        assert "mu_daily" in holding
        assert holding["weight"] >= 0
