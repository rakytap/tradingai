from __future__ import annotations

from unittest.mock import patch

from tradingai.ibkr.buy_trail import (
    BuyTrailConfig,
    BuyTrailState,
    _trail_stop_from_activate,
    poll_once,
    start_buy_trail,
)


def test_trail_stop_from_activate_is_below_mid():
    stop = _trail_stop_from_activate(100.0, 0.6)
    assert stop < 100.0
    assert abs(stop - 99.4) < 0.02


def test_start_buy_trail_dry_run_skips_stp_and_plans_trail():
    captured: dict = {}

    def fake_buy(**kwargs):
        captured.update(kwargs)
        return {
            "submitted": False,
            "status": "dry_run",
            "quantity": 10,
            "limit_price": 50.0,
            "sizing": {"quantity": 10, "limit_price": 50.0},
        }

    with patch(
        "tradingai.ibkr.tools.orders.buy_stock_budget_tool",
        side_effect=fake_buy,
    ):
        result = start_buy_trail(
            symbol="AAA",
            total_budget=1000,
            limit_price=50.0,
            submit=False,
        )
    assert captured["stop_loss"] is None
    assert result.get("watcher") is None
    assert result["plan"]["activate_price"] == 50.0
    assert result["plan"]["initial_trail_percent"] == 0.6
    assert result["plan"]["tight_trail_percent"] == 0.05
    assert result["plan"]["trail_1_5x_percent"] == 0.03
    assert result["plan"]["trail_2x_percent"] == 0.01
    assert result["plan"]["tight_interval_sec"] == 5.0


def test_poll_steps_trail_and_exits_when_flat():
    cfg = BuyTrailConfig(
        symbol="AAA",
        buy_order_id=7,
        target_qty=10,
        activate_price=100.0,
        initial_trail_percent=0.6,
        tight_trail_percent=0.05,
        trail_1_5x_percent=0.03,
        trail_2x_percent=0.01,
        target_return=0.01,
        submit=True,
    )
    state = BuyTrailState(job_id="AAA_7", config={})

    def snapshot_fn(**kwargs):
        oid = kwargs["order_id"]
        if oid == 7:
            return {
                "order_id": 7,
                "open": False,
                "open_order": {"filled": 10, "avg_fill_price": 100.0},
                "filled_shares": 10,
                "fills": [
                    {
                        "exec_id": "e1",
                        "side": "buy",
                        "shares": 10,
                        "price": 100.0,
                        "order_id": 7,
                    }
                ],
            }
        return {"order_id": oid, "open": True, "open_order": {}, "filled_shares": 0}

    sells: list[dict] = []

    def sell_fn(**kwargs):
        sells.append(kwargs)
        return {"status": "Submitted", "submitted": True, "order_id": 99}

    mods: list[dict] = []

    def modify_fn(**kwargs):
        mods.append(kwargs)
        return {"submitted": True, "status": "Submitted"}

    pos_qty = {"q": 10.0}

    def position_fn(**kwargs):
        return {"quantity": pos_qty["q"]}

    # Fill → place trail
    state = poll_once(
        cfg,
        state,
        snapshot_fn=snapshot_fn,
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        mid_fn=lambda **kwargs: {"mid_price": 100.0},
        position_fn=position_fn,
    )
    assert state.buy_complete is True
    assert len(sells) == 1
    assert sells[0]["trail_percent"] == 0.6
    assert state.trail_order_id == 99
    assert state.trail_stage == 0
    assert state.target_mid is not None
    assert state.target_mid_1_5x is not None
    assert state.target_mid_2x is not None
    assert state.target_mid_2x > state.target_mid_1_5x > state.target_mid

    # Below 1× → no tighten
    state = poll_once(
        cfg,
        state,
        snapshot_fn=snapshot_fn,
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        mid_fn=lambda **kwargs: {"mid_price": 100.1},
        position_fn=position_fn,
    )
    assert len(mods) == 0
    assert state.trail_stage == 0

    # 1× → 0.05%
    state = poll_once(
        cfg,
        state,
        snapshot_fn=snapshot_fn,
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        mid_fn=lambda **kwargs: {"mid_price": float(state.target_mid) + 0.01},
        position_fn=position_fn,
    )
    assert mods[-1]["trail_percent"] == 0.05
    assert state.trail_stage == 1

    # 1.5× → 0.03%
    state = poll_once(
        cfg,
        state,
        snapshot_fn=snapshot_fn,
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        mid_fn=lambda **kwargs: {"mid_price": float(state.target_mid_1_5x) + 0.01},
        position_fn=position_fn,
    )
    assert mods[-1]["trail_percent"] == 0.03
    assert state.trail_stage == 2

    # 2× → 0.01%
    state = poll_once(
        cfg,
        state,
        snapshot_fn=snapshot_fn,
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        mid_fn=lambda **kwargs: {"mid_price": float(state.target_mid_2x) + 0.01},
        position_fn=position_fn,
    )
    assert mods[-1]["trail_percent"] == 0.01
    assert state.trail_stage == 3

    # Position sold → complete
    pos_qty["q"] = 0.0
    state = poll_once(
        cfg,
        state,
        snapshot_fn=snapshot_fn,
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        mid_fn=lambda **kwargs: {"mid_price": float(state.target_mid_2x) + 1.0},
        position_fn=position_fn,
    )
    assert state.status == "completed"
    assert state.phase == "completed"
    assert any(a.get("event") == "position_depleted" for a in state.actions)
