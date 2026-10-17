from __future__ import annotations

import pytest

from tradingai.intraday.profit_target import (
    DEFAULT_BUY_NOTIONAL_USD,
    DEFAULT_PROFIT_TAKER_BUFFER,
    DEFAULT_TARGET_RETURN,
    fee_aware_sell_price,
    sell_trail_pct,
    target_sell_price,
    trail_stop_price,
)
from tradingai.predict.ibkr_fees import estimate_order_commission


def test_default_target_return_formula():
    assert DEFAULT_TARGET_RETURN == pytest.approx((1.0 / 365.0) * (7.0 / 5.0))
    assert DEFAULT_BUY_NOTIONAL_USD == 20_000.0
    assert DEFAULT_PROFIT_TAKER_BUFFER == pytest.approx(0.0002)


def test_sell_trail_defaults_to_half_target_return():
    trail = sell_trail_pct()
    assert trail["sell_trail"] == pytest.approx(DEFAULT_TARGET_RETURN / 2.0)
    assert trail["sell_trail_pct"] == pytest.approx(DEFAULT_TARGET_RETURN / 2.0 * 100.0)
    custom = sell_trail_pct(sell_trail=0.01)
    assert custom["sell_trail"] == pytest.approx(0.01)
    assert custom["sell_trail_pct"] == pytest.approx(1.0)
    assert trail_stop_price(100.0, sell_trail=DEFAULT_TARGET_RETURN / 2.0) == pytest.approx(
        100.0 * (1.0 - DEFAULT_TARGET_RETURN / 2.0)
    )


def test_fee_aware_sell_price_nets_requested_return():
    buy_price = 50.0
    notional = 20_000.0
    ret = DEFAULT_TARGET_RETURN
    result = fee_aware_sell_price(
        buy_price,
        buy_notional_usd=notional,
        net_return=ret,
    )
    shares = notional / buy_price
    sell_value = shares * result["sell_price"]
    buy_fee = estimate_order_commission(shares, notional)
    sell_fee = estimate_order_commission(shares, sell_value, monthly_volume_shares=shares)
    net_profit = sell_value - sell_fee - notional - buy_fee
    assert net_profit == pytest.approx(ret * notional, rel=1e-4, abs=0.02)


def test_target_sell_price_profit_taker_and_trail():
    buy_price = 50.0
    notional = 20_000.0
    ret = DEFAULT_TARGET_RETURN
    result = target_sell_price(
        buy_price,
        buy_notional_usd=notional,
        target_return=ret,
    )
    shares = notional / buy_price
    taker_return = ret + DEFAULT_PROFIT_TAKER_BUFFER
    sell_value = shares * result["profit_taker_price"]
    buy_fee = estimate_order_commission(shares, notional)
    sell_fee = estimate_order_commission(shares, sell_value, monthly_volume_shares=shares)
    net_profit = sell_value - sell_fee - notional - buy_fee
    assert net_profit == pytest.approx(taker_return * notional, rel=1e-4, abs=0.02)
    assert result["profit_taker_price"] == result["target_sell_price"]
    assert result["profit_taker_return"] == pytest.approx(taker_return)
    assert result["trail_activate_return"] == pytest.approx(ret / 2.0)
    assert result["sell_trail"] == pytest.approx(ret / 2.0)
    assert result["trail_activate_price"] < result["profit_taker_price"]
    assert result["trail_activate_price"] > buy_price
    assert result["trail_stop_at_activation_price"] == pytest.approx(
        result["trail_activate_price"] * (1.0 - ret / 2.0)
    )

    # Trail activation nets half target after fees.
    act = fee_aware_sell_price(
        buy_price, buy_notional_usd=notional, net_return=ret / 2.0
    )
    assert result["trail_activate_price"] == pytest.approx(act["sell_price"], abs=1e-6)


def test_larger_notional_changes_share_count():
    a = target_sell_price(100.0, buy_notional_usd=10_000.0)
    b = target_sell_price(100.0, buy_notional_usd=40_000.0)
    assert a["shares"] == pytest.approx(100.0)
    assert b["shares"] == pytest.approx(400.0)
    assert b["profit_taker_price"] > 0
