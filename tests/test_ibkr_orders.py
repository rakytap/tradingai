from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingai.providers import ibkr_tws


def test_place_equity_order_dry_run():
    out = ibkr_tws.place_equity_order(
        symbol="AAPL",
        side="buy",
        quantity=10,
        dry_run=True,
    )
    assert out["submitted"] is False
    assert out["status"] == "dry_run"
    assert out["side"] == "buy"
    assert out["quantity"] == 10


def test_place_equity_order_requires_allow_flag():
    with pytest.raises(PermissionError):
        ibkr_tws.place_equity_order(
            symbol="AAPL",
            side="buy",
            quantity=1,
            dry_run=False,
            allow_orders=False,
        )


def test_place_equity_order_limit_requires_price():
    with pytest.raises(ValueError, match="limit_price"):
        ibkr_tws.place_equity_order(
            symbol="AAPL",
            side="sell",
            quantity=1,
            order_type="LMT",
            dry_run=True,
        )


def test_place_equity_order_max_shares():
    with pytest.raises(ValueError, match="max_shares"):
        ibkr_tws.place_equity_order(
            symbol="AAPL",
            side="buy",
            quantity=50,
            max_shares=10,
            dry_run=True,
        )


def test_integer_shares_for_budget():
    assert ibkr_tws.integer_shares_for_budget(19000, 100) == 190
    assert ibkr_tws.integer_shares_for_budget(99.9, 100) == 0
    assert ibkr_tws.integer_shares_for_budget(20000 * 0.95, 437.6) == 43


def test_place_budget_limit_buy_dry_run_with_price():
    out = ibkr_tws.place_budget_limit_buy(
        symbol="AAPL",
        total_budget=20_000,
        limit_price=100.0,
        budget_fraction=0.95,
        dry_run=True,
    )
    assert out["submitted"] is False
    assert out["status"] == "dry_run"
    assert out["order_type"] == "LMT"
    assert out["limit_price"] == 100.0
    assert out["quantity"] == 190  # floor(19000/100)
    assert out["sizing"]["spend_budget"] == 19000.0
    assert out["sizing"]["notional"] == 19000.0
    assert out["stop_loss"] == pytest.approx(0.01)
    assert out["stop_price"] == pytest.approx(99.0)
    assert out["sizing"]["stop_price"] == pytest.approx(99.0)


def test_place_budget_limit_buy_no_stop_loss():
    out = ibkr_tws.place_budget_limit_buy(
        symbol="AAPL",
        total_budget=20_000,
        limit_price=100.0,
        budget_fraction=0.95,
        stop_loss=None,
        dry_run=True,
    )
    assert out["submitted"] is False
    assert out["status"] == "dry_run"
    assert out["quantity"] == 190
    assert out["sizing"]["stop_loss"] is None
    assert out["sizing"]["stop_price"] is None


def test_place_budget_limit_buy_custom_stop_loss():
    out = ibkr_tws.place_budget_limit_buy(
        symbol="COIN",
        total_budget=6_000,
        limit_price=170.0,
        budget_fraction=0.95,
        stop_loss=0.02,
        dry_run=True,
    )
    assert out["quantity"] == 33  # floor(5700/170)
    assert out["stop_loss"] == pytest.approx(0.02)
    assert out["stop_price"] == pytest.approx(166.6)


def test_place_budget_limit_buy_rejects_tiny_budget():
    out = ibkr_tws.place_budget_limit_buy(
        symbol="AAPL",
        total_budget=50,
        limit_price=100.0,
        dry_run=True,
    )
    assert out["submitted"] is False
    assert out["status"] == "rejected"
    assert out["quantity"] == 0
    assert "Budget too small" in out["error"]


def test_place_budget_limit_buy_uses_mid_when_price_omitted(monkeypatch):
    monkeypatch.setattr(
        ibkr_tws,
        "fetch_equity_mid_price",
        lambda symbol, **kwargs: {
            "symbol": symbol.upper(),
            "mid_price": 50.0,
            "price_source": "mid",
            "bid": 49.9,
            "ask": 50.1,
            "last": 50.0,
            "close": 49.0,
            "source": "ibkr_tws",
        },
    )
    out = ibkr_tws.place_budget_limit_buy(
        symbol="msft",
        total_budget=10_000,
        limit_price=None,
        budget_fraction=0.95,
        dry_run=True,
    )
    assert out["limit_price"] == 50.0
    assert out["quantity"] == 190  # floor(9500/50)
    assert out["quote"]["price_source"] == "mid"
    assert out["sizing"]["limit_price"] == 50.0
    assert out["stop_price"] == pytest.approx(49.5)


def test_round_us_equity_price():
    assert ibkr_tws.round_us_equity_price(843.4552507733332, mode="ceil") == 843.46
    assert ibkr_tws.round_us_equity_price(843.4552507733332, mode="round") == 843.46
    assert ibkr_tws.round_us_equity_price(843.454, mode="floor") == 843.45
    assert ibkr_tws.round_us_equity_price(0.12346, mode="round") == 0.1235  # $0.0001 tick
    assert ibkr_tws.round_us_equity_offset(0.42173) == 0.42
    assert ibkr_tws.round_us_equity_offset(0.085325) == 0.09


def test_place_trail_limit_sell_dry_run_explicit():
    out = ibkr_tws.place_trail_limit_sell(
        symbol="COIN",
        quantity=33,
        limit_price=170.65,
        trail_percent=0.05,
        dry_run=True,
    )
    assert out["submitted"] is False
    assert out["status"] == "dry_run"
    assert out["order_type"] == "TRAIL LIMIT"
    assert out["side"] == "sell"
    assert out["quantity"] == 33
    assert out["limit_price"] == pytest.approx(170.65)
    assert out["trail_percent"] == pytest.approx(0.05)
    # 170.65 * 0.05% = 0.085325 → round to $0.09 tick
    assert out["lmt_price_offset"] == pytest.approx(0.09)


def test_place_trail_limit_sell_rounds_fee_aware_target(monkeypatch):
    """GEV-style sub-penny fee target must be ceiled to a valid cent tick."""
    monkeypatch.setattr(
        ibkr_tws,
        "fetch_equity_position",
        lambda symbol, **kwargs: {
            "symbol": symbol.upper(),
            "account": "DU123",
            "quantity": 3.0,
            "avg_cost": 840.0,
            "sec_type": "STK",
            "source": "ibkr_tws",
        },
    )
    monkeypatch.setattr(
        ibkr_tws,
        "fetch_last_buy_price",
        lambda symbol, **kwargs: {
            "symbol": symbol.upper(),
            "buy_price": 840.0,
            "shares": 3.0,
            "time": None,
            "account": "DU123",
            "source": "ibkr_tws_execution",
        },
    )
    out = ibkr_tws.place_trail_limit_sell(
        symbol="GEV",
        quantity=None,
        limit_price=None,
        trail_percent=0.05,
        dry_run=True,
        target_return=0.003835616,
    )
    raw = out["sizing"]["target_meta"]["sell_price"]
    assert raw == pytest.approx(843.4552507733332)
    assert out["trail_stop_price"] == 843.46
    assert out["limit_price"] == 843.46
    assert out["lmt_price_offset"] == pytest.approx(0.42)


def test_place_trail_limit_sell_custom_target_return(monkeypatch):
    monkeypatch.setattr(
        ibkr_tws,
        "fetch_equity_position",
        lambda symbol, **kwargs: {
            "symbol": symbol.upper(),
            "account": "DU123",
            "quantity": 10.0,
            "avg_cost": 100.0,
            "sec_type": "STK",
            "source": "ibkr_tws",
        },
    )
    monkeypatch.setattr(
        ibkr_tws,
        "fetch_last_buy_price",
        lambda symbol, **kwargs: {
            "symbol": symbol.upper(),
            "buy_price": 100.0,
            "shares": 10.0,
            "time": None,
            "account": "DU123",
            "source": "ibkr_tws_execution",
        },
    )
    low = ibkr_tws.place_trail_limit_sell(
        symbol="AAA",
        limit_price=None,
        dry_run=True,
        target_return=0.005,
    )
    high = ibkr_tws.place_trail_limit_sell(
        symbol="AAA",
        limit_price=None,
        dry_run=True,
        target_return=0.01,
    )
    assert low["sizing"]["target_return"] == pytest.approx(0.005)
    assert high["sizing"]["target_return"] == pytest.approx(0.01)
    assert high["trail_stop_price"] > low["trail_stop_price"]


def test_place_trail_limit_sell_defaults_qty_and_target(monkeypatch):
    monkeypatch.setattr(
        ibkr_tws,
        "fetch_equity_position",
        lambda symbol, **kwargs: {
            "symbol": symbol.upper(),
            "account": "DU123",
            "quantity": 33.7,
            "avg_cost": 170.0,
            "sec_type": "STK",
            "source": "ibkr_tws",
        },
    )
    monkeypatch.setattr(
        ibkr_tws,
        "fetch_last_buy_price",
        lambda symbol, **kwargs: {
            "symbol": symbol.upper(),
            "buy_price": 170.0,
            "shares": 33.0,
            "time": None,
            "account": "DU123",
            "source": "ibkr_tws_execution",
        },
    )
    out = ibkr_tws.place_trail_limit_sell(
        symbol="COIN",
        quantity=None,
        limit_price=None,
        trail_percent=0.05,
        dry_run=True,
        target_return=0.003835616,
    )
    assert out["quantity"] == 33  # floor(33.7)
    assert out["order_type"] == "TRAIL LIMIT"
    assert out["limit_price"] > 170.0
    # Fee-aware target rounded up to a valid cent.
    assert abs(out["limit_price"] * 100 - round(out["limit_price"] * 100)) < 1e-9
    assert out["sizing"]["buy_price_source"] == "last_buy_fill"
    assert out["sizing"]["target_meta"] is not None


def test_place_trail_limit_sell_rejects_no_position(monkeypatch):
    monkeypatch.setattr(ibkr_tws, "fetch_equity_position", lambda symbol, **kwargs: None)
    monkeypatch.setattr(ibkr_tws, "fetch_last_buy_price", lambda symbol, **kwargs: None)
    out = ibkr_tws.place_trail_limit_sell(
        symbol="ZZZZ",
        quantity=None,
        limit_price=100.0,
        dry_run=True,
    )
    assert out["status"] == "rejected"
    assert "No open long position" in out["error"]


def test_place_trail_limit_sell_submits(monkeypatch):
    class FakeOrderStatus:
        status = "Submitted"
        filled = 0.0
        remaining = 10.0
        avgFillPrice = 0.0
        whyHeld = ""

    class FakeTrade:
        def __init__(self):
            self.order = SimpleNamespace(orderId=7, permId=8)
            self.orderStatus = FakeOrderStatus()
            self.fills = []
            self.log = ["submitted"]

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def placeOrder(self, contract, order):
            assert order.action == "SELL"
            assert order.orderType == "TRAIL LIMIT"
            assert order.totalQuantity == 10
            assert order.trailingPercent == 0.05
            assert order.trailStopPrice == 200.0
            assert order.lmtPriceOffset == 0.1  # 200 * 0.05% → $0.10
            return FakeTrade()

        def sleep(self, seconds):
            return None

    class FakeCtx:
        def __enter__(self):
            return FakeIB()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(ibkr_tws, "connect_ib", lambda **kwargs: FakeCtx())
    out = ibkr_tws.place_trail_limit_sell(
        symbol="AAPL",
        quantity=10,
        limit_price=200.0,
        trail_percent=0.05,
        dry_run=False,
        allow_orders=True,
        wait_seconds=0.1,
    )
    assert out["submitted"] is True
    assert out["order_id"] == 7


def test_place_equity_order_submits(monkeypatch):
    class FakeOrderStatus:
        status = "Submitted"
        filled = 0.0
        remaining = 2.0
        avgFillPrice = 0.0
        whyHeld = ""

    class FakeTrade:
        def __init__(self):
            self.order = SimpleNamespace(orderId=42, permId=99)
            self.orderStatus = FakeOrderStatus()
            self.fills = []
            self.log = ["submitted"]

    class FakeIB:
        def qualifyContracts(self, contract):
            return [contract]

        def placeOrder(self, contract, order):
            assert order.action == "BUY"
            assert order.totalQuantity == 2
            return FakeTrade()

        def sleep(self, seconds):
            return None

    class FakeCtx:
        def __enter__(self):
            return FakeIB()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        ibkr_tws,
        "connect_ib",
        lambda **kwargs: FakeCtx(),
    )

    out = ibkr_tws.place_equity_order(
        symbol="MSFT",
        side="buy",
        quantity=2,
        dry_run=False,
        allow_orders=True,
        wait_seconds=0.1,
    )
    assert out["submitted"] is True
    assert out["order_id"] == 42
    assert out["status"] == "Submitted"
