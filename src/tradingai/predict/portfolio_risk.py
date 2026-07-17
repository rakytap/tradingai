from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from tradingai.analysis.ml.ohlcv import load_ticker_ohlcv

DEFAULT_LOOKBACK_DAYS = 126
DEFAULT_MIN_OBS = 60
DEFAULT_RISK_AVERSION = 3.0
DEFAULT_MAX_WEIGHT = 0.25
TRADING_DAYS_PER_YEAR = 252


def horizon_return_to_daily_mu(expected_return_pct: float, horizon_days: int) -> float:
    """Convert horizon expected return (%) to an equivalent daily mean return."""
    horizon = max(int(horizon_days), 1)
    total = 1.0 + float(expected_return_pct) / 100.0
    if total <= 0:
        return float((1e-6) ** (1.0 / horizon) - 1.0)
    return float(total ** (1.0 / horizon) - 1.0)


def build_returns_frame(
    tickers: list[str],
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_obs: int = DEFAULT_MIN_OBS,
    database_url: str | None = None,
) -> pd.DataFrame:
    """Build overlapping daily return columns from synced daily_prices."""
    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        normalized = ticker.upper().strip()
        ohlcv = load_ticker_ohlcv(normalized, database_url=database_url)
        if ohlcv.empty or "close" not in ohlcv.columns:
            continue
        closes = ohlcv["close"].astype(float).tail(lookback_days + 1)
        returns = closes.pct_change().dropna()
        if len(returns) < min_obs:
            continue
        series_map[normalized] = returns.rename(normalized)

    if len(series_map) < 2:
        raise ValueError(
            "Need price history for at least 2 selected tickers to run Markowitz. "
            f"Found usable series for {sorted(series_map)}. Run tradingai-sync-prices."
        )

    frame = pd.concat(series_map.values(), axis=1, join="inner").dropna()
    if len(frame) < min_obs:
        raise ValueError(
            f"Overlapping return history has only {len(frame)} rows "
            f"(need >= {min_obs}). Run tradingai-sync-prices with longer history."
        )
    if frame.shape[1] < 2:
        raise ValueError(
            "Fewer than 2 tickers remain after aligning overlapping returns. "
            "Run tradingai-sync-prices."
        )
    return frame


def estimate_mu_sigma(
    returns: pd.DataFrame,
    rows: list[dict],
    *,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Estimate daily mu from predictions and daily Sigma via Ledoit–Wolf."""
    tickers = list(returns.columns)
    by_ticker = {str(row["ticker"]).upper(): row for row in rows}
    mu = np.array(
        [
            horizon_return_to_daily_mu(
                float(by_ticker[ticker]["expected_return_pct"]),
                horizon_days,
            )
            for ticker in tickers
        ],
        dtype=float,
    )
    lw = LedoitWolf().fit(returns.to_numpy(dtype=float))
    sigma = np.asarray(lw.covariance_, dtype=float)
    return mu, sigma, tickers


def optimize_markowitz_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    invested_fraction: float,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> np.ndarray:
    """Long-only mean-variance weights summing to invested_fraction."""
    n = len(mu)
    if n == 0:
        return np.array([], dtype=float)

    invested = max(0.0, min(1.0, float(invested_fraction)))
    cap = min(float(max_weight), invested) if n == 1 else float(max_weight)
    if n * cap + 1e-12 < invested:
        # Small baskets cannot meet a tight per-name cap; relax to equal share.
        cap = invested / n

    x0 = np.full(n, invested / n, dtype=float)

    def objective(w: np.ndarray) -> float:
        return float(-(w @ mu) + 0.5 * risk_aversion * (w @ sigma @ w))

    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - invested)}
    bounds = [(0.0, min(cap, invested)) for _ in range(n)]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Markowitz optimization failed: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("Markowitz optimization produced zero weights.")
    weights *= invested / total
    return weights


def annualized_volatility_pct(sigma: np.ndarray, index: int) -> float:
    daily_var = float(sigma[index, index])
    return float(np.sqrt(max(daily_var, 0.0) * TRADING_DAYS_PER_YEAR) * 100.0)
