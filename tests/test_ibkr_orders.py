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
