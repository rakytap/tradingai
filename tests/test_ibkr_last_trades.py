from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from tradingai.providers import ibkr_tws

ET = ZoneInfo("America/New_York")


def _fill(*, side: str, symbol: str, shares: float, price: float, when: datetime, exec_id: str):
    return SimpleNamespace(
        time=when,
        contract=SimpleNamespace(
            symbol=symbol,
            localSymbol=symbol,
            secType="STK",
            exchange="SMART",
        ),
        execution=SimpleNamespace(
            execId=exec_id,
            time=when,
            acctNumber="DU123",
            exchange="NYSE",
            side=side,
            shares=shares,
            price=price,
            permId=1,
            clientId=7,
            orderId=10,
            liquidation=0,
            cumQty=shares,
            avgPrice=price,
            orderRef="",
            evRule="",
            evMultiplier=0.0,
            modelCode="",
            lastLiquidity=0,
        ),
        commissionReport=SimpleNamespace(
            commission=1.25,
            currency="USD",
            realizedPNL=12.5,
        ),
    )


def test_normalize_trade_side():
    assert ibkr_tws.normalize_trade_side("buy") == "buy"
    assert ibkr_tws.normalize_trade_side("BOT") == "buy"
    assert ibkr_tws.normalize_trade_side("sell") == "sell"
    assert ibkr_tws.normalize_trade_side("SLD") == "sell"
    assert ibkr_tws.normalize_trade_side("all") is None
    with pytest.raises(ValueError):
        ibkr_tws.normalize_trade_side("hold")


def test_serialize_fill_maps_bot_sld():
    fill = _fill(
        side="BOT",
        symbol="AAPL",
        shares=10,
        price=100.0,
        when=datetime(2026, 10, 20, 10, 0, tzinfo=ET),
        exec_id="1",
    )
    row = ibkr_tws.serialize_fill(fill)
    assert row["side"] == "buy"
    assert row["side_raw"] == "BOT"
    assert row["notional_usd"] == 1000.0
    assert row["commission"] == 1.25
    assert row["realized_pnl"] == 12.5


def test_fetch_recent_trades_filters_and_limits(monkeypatch):
    fills = [
        _fill(
            side="BOT",
            symbol="AAPL",
            shares=5,
            price=100.0,
            when=datetime(2026, 10, 20, 11, 0, tzinfo=ET),
            exec_id="a",
        ),
        _fill(
            side="SLD",
            symbol="AAPL",
            shares=2,
            price=101.0,
            when=datetime(2026, 10, 20, 12, 0, tzinfo=ET),
            exec_id="b",
        ),
        _fill(
            side="SLD",
            symbol="MSFT",
            shares=3,
            price=400.0,
            when=datetime(2026, 10, 20, 13, 0, tzinfo=ET),
            exec_id="c",
        ),
    ]

    class FakeIB:
        def reqExecutions(self, exec_filter):
            return list(fills)

    class FakeCtx:
        def __enter__(self):
            return FakeIB()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(ibkr_tws, "connect_ib", lambda **kwargs: FakeCtx())

    all_trades = ibkr_tws.fetch_recent_trades(limit=10, side="all")
    assert all_trades["count"] == 3
    assert all_trades["trades"][0]["exec_id"] == "c"

    sells = ibkr_tws.fetch_recent_trades(limit=10, side="sell")
    assert sells["count"] == 2
    assert all(t["side"] == "sell" for t in sells["trades"])

    aapl_sells = ibkr_tws.fetch_recent_trades(limit=10, side="sell", symbol="AAPL")
    assert aapl_sells["count"] == 1
    assert aapl_sells["trades"][0]["symbol"] == "AAPL"

    limited = ibkr_tws.fetch_recent_trades(limit=1, side="all")
    assert limited["count"] == 1
    assert limited["available_before_limit"] == 3
