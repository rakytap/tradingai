from __future__ import annotations

from unittest.mock import patch

from tradingai.ibkr.sell_trail import (
    SellTrailConfig,
    SellTrailState,
    _trail_stop_from_mid,
    poll_once,
    start_sell_trail,
)


def test_trail_stop_from_mid_is_below_mid():
    stop = _trail_stop_from_mid(100.0, 0.05)
    assert stop < 100.0
    assert abs(stop - 99.95) < 0.02


def test_start_sell_trail_dry_run_plans_without_watcher():
    def fake_resolve(symbol, **kwargs):
        return {
            "symbol": symbol.upper(),
            "quantity": 10.0,
            "position_quantity": 10.0,
            "buy_price": 100.0,
            "buy_price_source": "last_buy_fill",
            "position": {"quantity": 10.0, "avg_cost": 100.0},
            "last_buy": {"buy_price": 100.0},
        }

    with patch(
        "tradingai.ibkr.sell_trail.resolve_position_and_buy",
        side_effect=fake_resolve,
    ):
        result = start_sell_trail(symbol="AAA", submit=False)

    assert result.get("watcher") is None
    plan = result["plan"]
    assert plan["buy_price"] == 100.0
    assert plan["activate_return"] == 0.002
    assert plan["initial_trail_percent"] == 0.05
    assert plan["trail_1_5x_percent"] == 0.03
    assert plan["trail_2x_percent"] == 0.01
    assert plan["interval_sec"] == 10.0
    assert plan["activate_mid"] is not None
    assert plan["mid_2x"] > plan["mid_1_5x"] > plan["activate_mid"]


def test_poll_activates_steps_and_exits_on_final_or_flat():
    cfg = SellTrailConfig(
        symbol="AAA",
        quantity=10.0,
        buy_price=100.0,
        activate_return=0.002,
        initial_trail_percent=0.05,
        trail_1_5x_percent=0.03,
        trail_2x_percent=0.01,
        submit=True,
        interval_sec=10.0,
    )
    state = SellTrailState(job_id="AAA_t", config={})

    # Seed thresholds without fee math variability.
    state.activate_mid = 100.25
    state.mid_1_5x = 100.35
    state.mid_2x = 100.45

    sells: list[dict] = []
    mods: list[dict] = []
    pos_qty = {"q": 10.0}

    def sell_fn(**kwargs):
        sells.append(kwargs)
        return {"status": "Submitted", "submitted": True, "order_id": 55}

    def modify_fn(**kwargs):
        mods.append(kwargs)
        return {"submitted": True, "status": "Submitted"}

    def position_fn(**kwargs):
        return {"quantity": pos_qty["q"]}

    # Below activate → wait
    state = poll_once(
        cfg,
        state,
        mid_fn=lambda **kwargs: {"mid_price": 100.10},
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        position_fn=position_fn,
    )
    assert len(sells) == 0
    assert state.phase == "waiting_activate"
    assert state.trail_stage == 0

    # Hit activate → place 0.05% trail
    state = poll_once(
        cfg,
        state,
        mid_fn=lambda **kwargs: {"mid_price": 100.26},
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        position_fn=position_fn,
    )
    assert len(sells) == 1
    assert sells[0]["trail_percent"] == 0.05
    assert state.trail_order_id == 55
    assert state.trail_stage == 1
    assert state.phase == "trail_active"

    # 1.5× → 0.03%
    state = poll_once(
        cfg,
        state,
        mid_fn=lambda **kwargs: {"mid_price": 100.36},
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        position_fn=position_fn,
    )
    assert mods[-1]["trail_percent"] == 0.03
    assert state.trail_stage == 2

    # 2× → 0.01% and complete
    state = poll_once(
        cfg,
        state,
        mid_fn=lambda **kwargs: {"mid_price": 100.46},
        sell_fn=sell_fn,
        modify_fn=modify_fn,
        position_fn=position_fn,
    )
    assert mods[-1]["trail_percent"] == 0.01
    assert state.trail_stage == 3
    assert state.status == "completed"
    assert any(a.get("event") == "final_trail_applied" for a in state.actions)


def test_poll_exits_when_position_flat():
    cfg = SellTrailConfig(
        symbol="AAA",
        quantity=10.0,
        buy_price=100.0,
        submit=True,
    )
    state = SellTrailState(
        job_id="AAA_flat",
        config={},
        trail_order_id=9,
        trail_stage=1,
        activate_mid=100.25,
        mid_1_5x=100.35,
        mid_2x=100.45,
    )

    state = poll_once(
        cfg,
        state,
        mid_fn=lambda **kwargs: {"mid_price": 100.30},
        sell_fn=lambda **kwargs: {},
        modify_fn=lambda **kwargs: {},
        position_fn=lambda **kwargs: {"quantity": 0.0},
    )
    assert state.status == "completed"
    assert any(a.get("event") == "position_depleted" for a in state.actions)
