"""IBKR Pro Tiered US stock/ETF broker commission estimates.

Broker commission only (no exchange / regulatory / clearing pass-through).
Rates match Interactive Brokers published Tiered US equity pricing.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# (monthly_share_volume_upper_inclusive, USD per share). Last upper is effectively unbounded.
TIERED_US_RATES: tuple[tuple[float, float], ...] = (
    (300_000, 0.0035),
    (3_000_000, 0.0020),
    (20_000_000, 0.0015),
    (100_000_000, 0.0010),
    (float("inf"), 0.0005),
)

MIN_ORDER_USD = 0.35
MAX_TRADE_VALUE_FRAC = 0.005  # Tiered commissions capped at 0.5% of trade value


def tier_rate_per_share(monthly_volume_shares: float = 0.0) -> float:
    """Marginal Tiered USD/share rate given month-to-date share volume before this trade."""
    volume = max(0.0, float(monthly_volume_shares))
    for upper, rate in TIERED_US_RATES:
        if volume <= upper:
            return float(rate)
    return float(TIERED_US_RATES[-1][1])


def min_economic_trade_usd() -> float:
    """Smallest trade value at which the $0.35 floor can bind (else 0.5% cap wins)."""
    return float(MIN_ORDER_USD / MAX_TRADE_VALUE_FRAC)


def estimate_order_commission(
    shares: float,
    trade_value_usd: float,
    *,
    monthly_volume_shares: float = 0.0,
) -> float:
    """Estimate IBKR Pro Tiered broker commission for one order.

    Commission = min(max(per_share, $0.35), 0.5% of trade value).
    When the percentage max is below the minimum, the max wins (IBKR rule).
    """
    qty = abs(float(shares))
    value = abs(float(trade_value_usd))
    if qty <= 0.0 or value <= 0.0:
        return 0.0

    rate = tier_rate_per_share(monthly_volume_shares)
    raw = rate * qty
    floored = max(MIN_ORDER_USD, raw)
    return float(min(floored, MAX_TRADE_VALUE_FRAC * value))


def estimate_rebalance_commission(
    trades: Sequence[Mapping[str, float]],
    *,
    monthly_volume_shares: float = 0.0,
) -> float:
    """Sum Tiered commissions over a list of trades.

    Each trade mapping must include ``shares`` and ``trade_value_usd`` (or ``value``).
    Monthly volume is advanced by traded shares so later names can move into lower tiers.
    """
    volume = max(0.0, float(monthly_volume_shares))
    total = 0.0
    for trade in trades:
        shares = abs(float(trade.get("shares", 0.0)))
        value = abs(
            float(
                trade["trade_value_usd"]
                if "trade_value_usd" in trade
                else trade.get("value", 0.0)
            )
        )
        fee = estimate_order_commission(
            shares,
            value,
            monthly_volume_shares=volume,
        )
        total += fee
        volume += shares
    return float(total)


def estimate_rebalance_commission_from_weights(
    *,
    tickers: Iterable[str],
    prior_weights: Mapping[str, float],
    new_weights: Mapping[str, float],
    prices: Mapping[str, float],
    notional_usd: float,
    monthly_volume_shares: float = 0.0,
) -> tuple[float, list[dict]]:
    """Build per-name trade fees from weight deltas and return (total, details)."""
    notional = float(notional_usd)
    details: list[dict] = []
    trades: list[dict] = []
    for ticker in tickers:
        key = str(ticker).upper()
        w0 = float(prior_weights.get(key, 0.0))
        w1 = float(new_weights.get(key, 0.0))
        delta_w = w1 - w0
        trade_usd = abs(delta_w) * notional
        price = float(prices.get(key, 0.0))
        shares = (trade_usd / price) if price > 0 else 0.0
        trades.append({"shares": shares, "trade_value_usd": trade_usd})
        details.append(
            {
                "ticker": key,
                "prior_weight": w0,
                "weight": w1,
                "trade_usd": round(trade_usd, 2),
                "shares": shares,
            }
        )

    total = estimate_rebalance_commission(
        trades,
        monthly_volume_shares=monthly_volume_shares,
    )
    # Attach per-order fees in the same sequential tier order.
    volume = max(0.0, float(monthly_volume_shares))
    for detail, trade in zip(details, trades):
        fee = estimate_order_commission(
            trade["shares"],
            trade["trade_value_usd"],
            monthly_volume_shares=volume,
        )
        detail["est_commission_usd"] = round(fee, 4)
        volume += abs(float(trade["shares"]))
    return float(total), details
