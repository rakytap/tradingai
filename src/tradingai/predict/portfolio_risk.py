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


def _soft_tiered_cost_frac(
    turnover: np.ndarray,
    prices: np.ndarray,
    *,
    notional: float,
    rate_per_share: float,
    min_order_usd: float,
    max_trade_value_frac: float,
) -> float:
    """Differentiable IBKR Tiered fee / notional for weight turnovers ``t``."""
    # Smooth max(a,b) / min(a,b) so SLSQP can feel the $0.35 floor.
    eps = 1e-12
    trade_usd = notional * turnover
    shares = trade_usd / prices
    raw = rate_per_share * shares
    floored = 0.5 * ((min_order_usd + raw) + np.sqrt((raw - min_order_usd) ** 2 + eps))
    cap = max_trade_value_frac * trade_usd
    fee = 0.5 * ((floored + cap) - np.sqrt((floored - cap) ** 2 + eps))
    # No fee when turnover is numerically zero.
    fee = np.where(turnover <= 1e-12, 0.0, fee)
    return float(fee.sum() / notional)


def suppress_uneconomic_rebalance_trades(
    weights: np.ndarray,
    prior_weights: np.ndarray,
    *,
    notional_usd: float,
    invested_fraction: float,
    commission_preference: float,
    min_trade_usd: float | None = None,
) -> np.ndarray:
    """Snap sub-minimum tickets back to prior when commission preference is active.

    Trades below ``MIN_ORDER / 0.5%`` never pay the full $0.35 floor (cap wins) and
    are noise relative to IBKR's minimum; with ``gamma > 0`` keep the prior leg.
    """
    from tradingai.predict.ibkr_fees import min_economic_trade_usd

    gamma = max(0.0, float(commission_preference))
    if gamma <= 0.0:
        return np.asarray(weights, dtype=float)

    threshold = float(min_trade_usd) if min_trade_usd is not None else min_economic_trade_usd()
    # Stronger preference => stricter ticket size (full economic min at gamma>=1).
    threshold *= min(1.0, gamma)

    notional = max(float(notional_usd), 1e-9)
    invested = max(0.0, min(1.0, float(invested_fraction)))
    w = np.asarray(weights, dtype=float).copy()
    w0 = np.asarray(prior_weights, dtype=float)
    trade_usd = np.abs(w - w0) * notional
    uneconomic = (trade_usd > 1e-9) & (trade_usd < threshold)
    if not np.any(uneconomic):
        return w

    w[uneconomic] = w0[uneconomic]
    # Renormalize the movable sleeve so invested weight is preserved.
    free = ~uneconomic
    target_free = invested - float(w[~free].sum())
    free_sum = float(w[free].sum())
    if free_sum > 1e-12 and target_free > 0:
        w[free] *= target_free / free_sum
    elif target_free <= 1e-12:
        w = w0 * (invested / max(float(w0.sum()), 1e-12))
    total = float(w.sum())
    if total > 0:
        w *= invested / total
    return w


def optimize_markowitz_rebalance_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    prior_weights: np.ndarray,
    prices: np.ndarray,
    *,
    invested_fraction: float,
    notional_usd: float,
    commission_preference: float,
    rate_per_share: float,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> np.ndarray:
    """Long-only mean-variance with soft IBKR Tiered cost on turnover.

    Decision vector is ``[w_0..w_{n-1}, t_0..t_{n-1}]`` where ``t_i >= |w_i - w0_i|``.
    Soft cost approximates Tiered ``min(max(per_share, $0.35), 0.5% TV)`` / notional.
    A post-pass snaps tickets below the economic minimum when ``gamma > 0``.
    """
    from tradingai.predict.ibkr_fees import MAX_TRADE_VALUE_FRAC, MIN_ORDER_USD

    n = len(mu)
    if n == 0:
        return np.array([], dtype=float)
    if len(prior_weights) != n or len(prices) != n or sigma.shape != (n, n):
        raise ValueError("mu, sigma, prior_weights, and prices must align.")

    invested = max(0.0, min(1.0, float(invested_fraction)))
    notional = max(float(notional_usd), 1e-9)
    gamma = max(0.0, float(commission_preference))
    rate = max(0.0, float(rate_per_share))
    w0 = np.asarray(prior_weights, dtype=float).reshape(n)
    px = np.asarray(prices, dtype=float).reshape(n)
    if np.any(px <= 0):
        raise ValueError("All prices must be positive for rebalance optimization.")

    cap = min(float(max_weight), invested) if n == 1 else float(max_weight)
    if n * cap + 1e-12 < invested:
        cap = invested / n

    w_init = np.clip(w0, 0.0, cap)
    w_sum = float(w_init.sum())
    if w_sum <= 1e-12:
        w_init = np.full(n, invested / n, dtype=float)
    else:
        w_init = w_init * (invested / w_sum)
    t_init = np.abs(w_init - w0)
    x0 = np.concatenate([w_init, t_init])

    def objective(x: np.ndarray) -> float:
        w = x[:n]
        t = x[n:]
        mv = float(-(w @ mu) + 0.5 * risk_aversion * (w @ sigma @ w))
        if gamma <= 0.0:
            return mv
        soft_cost = gamma * _soft_tiered_cost_frac(
            t,
            px,
            notional=notional,
            rate_per_share=rate,
            min_order_usd=MIN_ORDER_USD,
            max_trade_value_frac=MAX_TRADE_VALUE_FRAC,
        )
        return mv + soft_cost

    constraints = [
        {"type": "eq", "fun": lambda x: float(np.sum(x[:n]) - invested)},
        *[
            {
                "type": "ineq",
                "fun": lambda x, i=i: float(x[n + i] - (x[i] - w0[i])),
            }
            for i in range(n)
        ],
        *[
            {
                "type": "ineq",
                "fun": lambda x, i=i: float(x[n + i] - (w0[i] - x[i])),
            }
            for i in range(n)
        ],
    ]
    bounds = [(0.0, min(cap, invested)) for _ in range(n)] + [
        (0.0, None) for _ in range(n)
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 800, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Markowitz rebalance optimization failed: {result.message}")

    weights = np.asarray(result.x[:n], dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("Markowitz rebalance produced zero weights.")
    weights *= invested / total
    return suppress_uneconomic_rebalance_trades(
        weights,
        w0,
        notional_usd=notional,
        invested_fraction=invested,
        commission_preference=gamma,
    )


def annualized_volatility_pct(sigma: np.ndarray, index: int) -> float:
    daily_var = float(sigma[index, index])
    return float(np.sqrt(max(daily_var, 0.0) * TRADING_DAYS_PER_YEAR) * 100.0)
