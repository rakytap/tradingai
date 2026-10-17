from __future__ import annotations

import pytest

from tradingai.predict.ibkr_fees import (
    MAX_TRADE_VALUE_FRAC,
    MIN_ORDER_USD,
    estimate_order_commission,
    estimate_rebalance_commission,
    tier_rate_per_share,
)


def test_tier_rate_first_and_higher_tiers():
    assert tier_rate_per_share(0) == pytest.approx(0.0035)
    assert tier_rate_per_share(300_000) == pytest.approx(0.0035)
    assert tier_rate_per_share(300_001) == pytest.approx(0.0020)
    assert tier_rate_per_share(50_000_000) == pytest.approx(0.0010)
    assert tier_rate_per_share(200_000_000) == pytest.approx(0.0005)


def test_order_commission_applies_minimum():
    # 10 shares @ $100 => raw = 0.035 < 0.35 min; value large enough for min.
    fee = estimate_order_commission(10, 1_000.0)
    assert fee == pytest.approx(MIN_ORDER_USD)


def test_order_commission_percentage_cap_beats_minimum_on_tiny_notional():
    # 10 shares @ $0.20 => max = 0.5% * 2 = 0.01 < min 0.35
    fee = estimate_order_commission(10, 2.0)
    assert fee == pytest.approx(MAX_TRADE_VALUE_FRAC * 2.0)


def test_order_commission_floor_then_cap_not_raw_per_share():
    # Tiny share count on a $5 ticket: raw << 0.5% TV << $0.35 → pay 0.5% TV.
    fee = estimate_order_commission(0.01, 5.0)
    assert fee == pytest.approx(MAX_TRADE_VALUE_FRAC * 5.0)
    assert fee > 0.0035 * 0.01


def test_order_commission_zero_on_empty_trade():
    assert estimate_order_commission(0, 100.0) == 0.0
    assert estimate_order_commission(10, 0.0) == 0.0


def test_rebalance_commission_advances_volume_across_trades():
    # First trade alone would be first-tier; after huge first trade, second uses lower rate.
    trades = [
        {"shares": 300_001, "trade_value_usd": 30_000_100.0},
        {"shares": 100, "trade_value_usd": 10_000.0},
    ]
    total = estimate_rebalance_commission(trades, monthly_volume_shares=0.0)
    second_only = estimate_order_commission(
        100,
        10_000.0,
        monthly_volume_shares=300_001,
    )
    first = estimate_order_commission(300_001, 30_000_100.0, monthly_volume_shares=0.0)
    assert total == pytest.approx(first + second_only)
