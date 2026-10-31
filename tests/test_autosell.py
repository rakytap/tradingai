from __future__ import annotations

from tradingai.ibkr.autosell import (
    AutosellConfig,
    AutosellState,
    find_unpaired_buys,
    poll_once,
)


def _t(side: str, shares: float, *, exec_id: str, time: str, price: float = 10.0):
    return {
        "exec_id": exec_id,
        "side": side,
        "shares": shares,
        "price": price,
        "time": time,
        "symbol": "AAA",
    }


def test_find_unpaired_buys_fifo():
    trades = [
        _t("buy", 10, exec_id="b1", time="2026-01-01T10:00:00+00:00"),
        _t("buy", 5, exec_id="b2", time="2026-01-01T10:01:00+00:00"),
        _t("sell", 10, exec_id="s1", time="2026-01-01T10:02:00+00:00"),
    ]
    unpaired = find_unpaired_buys(trades)
    assert len(unpaired) == 1
    assert unpaired[0]["exec_id"] == "b2"
    assert unpaired[0]["shares"] == 5


def test_find_unpaired_buys_partial_sell():
    trades = [
        _t("buy", 10, exec_id="b1", time="2026-01-01T10:00:00+00:00"),
        _t("sell", 4, exec_id="s1", time="2026-01-01T10:02:00+00:00"),
    ]
    unpaired = find_unpaired_buys(trades)
    assert len(unpaired) == 1
    assert unpaired[0]["shares"] == 6


def test_find_unpaired_buys_fully_matched():
    trades = [
        _t("buy", 3, exec_id="b1", time="2026-01-01T10:00:00+00:00"),
        _t("sell", 3, exec_id="s1", time="2026-01-01T10:02:00+00:00"),
    ]
    assert find_unpaired_buys(trades) == []


def test_poll_once_sells_unpaired_and_marks_handled():
    config = AutosellConfig(symbol="AAA", submit=False, trail_percent=0.05)
    state = AutosellState(job_id="AAA_test", config={})

    def fetch_trades(**kwargs):
        return {
            "trades": [
                _t("buy", 3, exec_id="b1", time="2026-01-01T10:00:00+00:00", price=100),
            ]
        }

    calls: list[dict] = []

    def sell_fn(**kwargs):
        calls.append(kwargs)
        return {
            "status": "dry_run",
            "submitted": False,
            "quantity": kwargs["quantity"],
        }

    state = poll_once(config, state, fetch_trades_fn=fetch_trades, sell_fn=sell_fn)
    assert len(calls) == 1
    assert calls[0]["symbol"] == "AAA"
    assert calls[0]["quantity"] == 3.0
    assert calls[0]["dry_run"] is True
    assert "b1" in state.handled_buy_keys
    assert state.sells_attempted == 1

    # Second poll should not re-fire.
    state = poll_once(config, state, fetch_trades_fn=fetch_trades, sell_fn=sell_fn)
    assert len(calls) == 1


def test_poll_once_respects_qty_cap():
    config = AutosellConfig(symbol="AAA", quantity=2, submit=True)
    state = AutosellState(job_id="AAA_cap", config={})

    def fetch_trades(**kwargs):
        return {
            "trades": [
                _t("buy", 10, exec_id="b9", time="2026-01-01T10:00:00+00:00"),
            ]
        }

    calls: list[dict] = []

    def sell_fn(**kwargs):
        calls.append(kwargs)
        return {"status": "Submitted", "submitted": True, "order_id": 1}

    poll_once(config, state, fetch_trades_fn=fetch_trades, sell_fn=sell_fn)
    assert calls[0]["quantity"] == 2.0
    assert calls[0]["dry_run"] is False
