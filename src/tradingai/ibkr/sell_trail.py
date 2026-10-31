"""Watch an open long: place trail at 0.2% profit, step 0.05→0.03→0.01%, then exit."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingai.intraday.profit_target import fee_aware_sell_price
from tradingai.providers.ibkr_tws import (
    DEFAULT_SELL_TRAIL_1_5X_PERCENT,
    DEFAULT_SELL_TRAIL_2X_PERCENT,
    DEFAULT_SELL_TRAIL_ACTIVATE_RETURN,
    DEFAULT_SELL_TRAIL_PERCENT,
)

DEFAULT_POLL_INTERVAL_SEC = 10.0


def sell_trail_state_dir() -> Path:
    raw = (os.getenv("TRADINGAI_SELL_TRAIL_DIR") or "").strip()
    if raw:
        path = Path(raw).expanduser()
    else:
        path = Path("data") / "sell_trail"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class SellTrailConfig:
    symbol: str
    quantity: float
    buy_price: float
    activate_return: float = DEFAULT_SELL_TRAIL_ACTIVATE_RETURN
    initial_trail_percent: float = DEFAULT_SELL_TRAIL_PERCENT
    trail_1_5x_percent: float = DEFAULT_SELL_TRAIL_1_5X_PERCENT
    trail_2x_percent: float = DEFAULT_SELL_TRAIL_2X_PERCENT
    tif: str = "DAY"
    outside_rth: bool = False
    account: str | None = None
    submit: bool = False
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
    buy_price_source: str | None = None

    def normalized_symbol(self) -> str:
        return self.symbol.upper().strip()


@dataclass
class SellTrailState:
    job_id: str
    config: dict
    pid: int | None = None
    # waiting_activate | trail_active | completed | stopped | error
    status: str = "starting"
    phase: str = "waiting_activate"
    started_at: str | None = None
    stopped_at: str | None = None
    last_poll_at: str | None = None
    last_error: str | None = None
    polls: int = 0
    last_mid: float | None = None
    trail_order_id: int | None = None
    trail_percent: float | None = None
    # 0=waiting, 1=0.05% placed @1×, 2=0.03% @1.5×, 3=0.01% @2× (done)
    trail_stage: int = 0
    activate_mid: float | None = None
    mid_1_5x: float | None = None
    mid_2x: float | None = None
    actions: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> SellTrailState:
        return cls(
            job_id=str(data["job_id"]),
            config=dict(data.get("config") or {}),
            pid=data.get("pid"),
            status=str(data.get("status") or "starting"),
            phase=str(data.get("phase") or "waiting_activate"),
            started_at=data.get("started_at"),
            stopped_at=data.get("stopped_at"),
            last_poll_at=data.get("last_poll_at"),
            last_error=data.get("last_error"),
            polls=int(data.get("polls") or 0),
            last_mid=(
                float(data["last_mid"]) if data.get("last_mid") is not None else None
            ),
            trail_order_id=(
                int(data["trail_order_id"])
                if data.get("trail_order_id") is not None
                else None
            ),
            trail_percent=(
                float(data["trail_percent"])
                if data.get("trail_percent") is not None
                else None
            ),
            trail_stage=int(data.get("trail_stage") or 0),
            activate_mid=(
                float(data["activate_mid"])
                if data.get("activate_mid") is not None
                else None
            ),
            mid_1_5x=(
                float(data["mid_1_5x"]) if data.get("mid_1_5x") is not None else None
            ),
            mid_2x=(
                float(data["mid_2x"]) if data.get("mid_2x") is not None else None
            ),
            actions=list(data.get("actions") or []),
        )


def job_paths(job_id: str) -> dict[str, Path]:
    root = sell_trail_state_dir() / job_id
    return {
        "dir": root,
        "state": root / "state.json",
        "log": root / "job.log",
        "pid": root / "job.pid",
    }


def make_job_id(symbol: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol.upper().strip()}_{stamp}"


def save_state(state: SellTrailState) -> Path:
    paths = job_paths(state.job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": state.job_id,
        "config": state.config,
        "pid": state.pid,
        "status": state.status,
        "phase": state.phase,
        "started_at": state.started_at,
        "stopped_at": state.stopped_at,
        "last_poll_at": state.last_poll_at,
        "last_error": state.last_error,
        "polls": state.polls,
        "last_mid": state.last_mid,
        "trail_order_id": state.trail_order_id,
        "trail_percent": state.trail_percent,
        "trail_stage": state.trail_stage,
        "activate_mid": state.activate_mid,
        "mid_1_5x": state.mid_1_5x,
        "mid_2x": state.mid_2x,
        "actions": state.actions[-200:],
    }
    paths["state"].write_text(json.dumps(payload, indent=2, default=str))
    if state.pid:
        paths["pid"].write_text(str(state.pid))
    return paths["state"]


def load_state(job_id: str) -> SellTrailState | None:
    path = job_paths(job_id)["state"]
    if not path.is_file():
        return None
    return SellTrailState.from_dict(json.loads(path.read_text()))


def list_jobs() -> list[dict]:
    root = sell_trail_state_dir()
    out: list[dict] = []
    for path in sorted(root.glob("*/state.json")):
        try:
            state = SellTrailState.from_dict(json.loads(path.read_text()))
        except Exception:
            continue
        cfg = state.config or {}
        out.append(
            {
                "job_id": state.job_id,
                "status": state.status,
                "phase": state.phase,
                "pid": state.pid,
                "alive": _pid_alive(state.pid),
                "symbol": cfg.get("symbol"),
                "buy_price": cfg.get("buy_price"),
                "quantity": cfg.get("quantity"),
                "trail_order_id": state.trail_order_id,
                "trail_percent": state.trail_percent,
                "trail_stage": state.trail_stage,
                "last_mid": state.last_mid,
                "activate_mid": state.activate_mid,
                "started_at": state.started_at,
                "log": str(job_paths(state.job_id)["log"]),
            }
        )
    return out


def job_status(job_id: str | None = None) -> dict:
    if job_id:
        state = load_state(job_id)
        if state is None:
            return {"error": f"unknown job_id {job_id}", "job_id": job_id}
        return {
            "job": {
                "job_id": state.job_id,
                "status": state.status,
                "phase": state.phase,
                "pid": state.pid,
                "alive": _pid_alive(state.pid),
                "config": state.config,
                "trail_order_id": state.trail_order_id,
                "trail_percent": state.trail_percent,
                "trail_stage": state.trail_stage,
                "last_mid": state.last_mid,
                "activate_mid": state.activate_mid,
                "mid_1_5x": state.mid_1_5x,
                "mid_2x": state.mid_2x,
                "last_error": state.last_error,
                "polls": state.polls,
                "actions": state.actions[-20:],
                "log": str(job_paths(state.job_id)["log"]),
            }
        }
    return {"jobs": list_jobs(), "state_dir": str(sell_trail_state_dir())}


def stop_job(job_id: str, *, force: bool = False) -> dict:
    state = load_state(job_id)
    if state is None:
        return {"error": f"unknown job_id {job_id}", "job_id": job_id}
    pid = state.pid
    if _pid_alive(pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception as exc:
            if force:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception as exc2:
                    return {"error": str(exc2), "job_id": job_id, "pid": pid}
            else:
                return {"error": str(exc), "job_id": job_id, "pid": pid}
    state.status = "stopped"
    state.phase = "stopped"
    state.stopped_at = _now_iso()
    save_state(state)
    return {
        "job_id": job_id,
        "status": "stopped",
        "pid": pid,
        "alive": _pid_alive(pid),
    }


def _append_action(state: SellTrailState, **kwargs: Any) -> None:
    row = {"at": _now_iso(), **kwargs}
    state.actions.append(row)


def _compute_profit_mid(*, buy_price: float, shares: float, net_return: float) -> float:
    meta = fee_aware_sell_price(
        float(buy_price),
        buy_notional_usd=float(buy_price) * float(shares),
        net_return=float(net_return),
    )
    return float(meta["sell_price"])


def _trail_stop_from_mid(mid: float, trail_percent: float) -> float:
    from tradingai.providers.ibkr_tws import round_us_equity_price

    stop = float(mid) * (1.0 - float(trail_percent) / 100.0)
    return round_us_equity_price(stop, mode="floor")


def _desired_stage(mid: float, state: SellTrailState) -> int:
    stage = 0
    if state.activate_mid is not None and mid + 1e-9 >= float(state.activate_mid):
        stage = 1
    if state.mid_1_5x is not None and mid + 1e-9 >= float(state.mid_1_5x):
        stage = 2
    if state.mid_2x is not None and mid + 1e-9 >= float(state.mid_2x):
        stage = 3
    return stage


def _trail_percent_for_stage(config: SellTrailConfig, stage: int) -> float:
    if stage >= 3:
        return float(config.trail_2x_percent)
    if stage == 2:
        return float(config.trail_1_5x_percent)
    return float(config.initial_trail_percent)


def _position_flat(position_fn, *, symbol: str, account: str | None) -> bool | None:
    try:
        pos = position_fn(symbol=symbol, account=account)
        return float(pos.get("quantity") or 0.0) <= 0.0
    except Exception:
        return None


def poll_once(
    config: SellTrailConfig,
    state: SellTrailState,
    *,
    mid_fn=None,
    sell_fn=None,
    modify_fn=None,
    position_fn=None,
) -> SellTrailState:
    """Wait for 0.2% profit → place 0.05% trail → 0.03% @1.5× → 0.01% @2× then exit."""
    from tradingai.ibkr.tools.orders import sell_stock_trail_tool
    from tradingai.intraday.config import get_ibkr_mcp_client_id
    from tradingai.providers.ibkr_tws import (
        fetch_equity_mid_price,
        fetch_equity_position,
        modify_trail_limit_percent,
    )

    cid = get_ibkr_mcp_client_id() + 60
    if mid_fn is None:

        def mid_fn(**kwargs):  # type: ignore[misc]
            return fetch_equity_mid_price(**kwargs, client_id=cid)

    if sell_fn is None:
        sell_fn = sell_stock_trail_tool

    if modify_fn is None:

        def modify_fn(**kwargs):  # type: ignore[misc]
            return modify_trail_limit_percent(**kwargs, client_id=cid + 1)

    if position_fn is None:

        def position_fn(**kwargs):  # type: ignore[misc]
            return fetch_equity_position(**kwargs, client_id=cid + 2)

    state.polls += 1
    state.last_poll_at = _now_iso()
    state.status = "running"
    sym = config.normalized_symbol()
    qty = float(int(config.quantity))
    buy_px = float(config.buy_price)
    act = float(config.activate_return)

    try:
        if state.activate_mid is None:
            state.activate_mid = _compute_profit_mid(
                buy_price=buy_px, shares=qty, net_return=act
            )
        if state.mid_1_5x is None:
            state.mid_1_5x = _compute_profit_mid(
                buy_price=buy_px, shares=qty, net_return=act * 1.5
            )
        if state.mid_2x is None:
            state.mid_2x = _compute_profit_mid(
                buy_price=buy_px, shares=qty, net_return=act * 2.0
            )

        flat = _position_flat(position_fn, symbol=sym, account=config.account)
        if flat is True:
            state.phase = "completed"
            state.status = "completed"
            state.last_error = None
            _append_action(state, event="position_depleted")
            return state

        mid_info = mid_fn(symbol=sym)
        mid = float(mid_info["mid_price"])
        state.last_mid = mid
        desired = _desired_stage(mid, state)

        # Still below activation profit — wait.
        if desired < 1:
            state.phase = "waiting_activate"
            return state

        # Place initial trail at 1× activation.
        if state.trail_order_id is None:
            trail_pct = float(config.initial_trail_percent)
            stop_px = _trail_stop_from_mid(mid, trail_pct)
            result = sell_fn(
                symbol=sym,
                quantity=qty,
                limit_price=stop_px,
                trail_percent=trail_pct,
                tif=config.tif,
                outside_rth=config.outside_rth,
                account=config.account,
                dry_run=not bool(config.submit),
            )
            ok = not (
                result.get("error")
                or result.get("status")
                in {
                    "rejected",
                    "Inactive",
                    "Cancelled",
                    "ApiCancelled",
                    "Rejected",
                }
            )
            _append_action(
                state,
                event="place_trail",
                trail_percent=trail_pct,
                trail_stop_price=stop_px,
                mid=mid,
                ok=ok,
                order_id=result.get("order_id"),
                status=result.get("status"),
                error=result.get("error"),
            )
            if ok and (result.get("submitted") or not config.submit):
                state.trail_order_id = (
                    int(result["order_id"])
                    if result.get("order_id") is not None
                    else None
                )
                state.trail_percent = trail_pct
                state.trail_stage = 1
                state.phase = "trail_active"
                state.last_error = None
                if not config.submit:
                    state.status = "completed"
                    state.phase = "completed"
            else:
                state.last_error = str(
                    result.get("error") or result.get("message") or "trail failed"
                )
                state.status = "error"
            return state

        # Step tighten 1.5× → 0.03%, 2× → 0.01%.
        if desired > int(state.trail_stage):
            stage = desired
            trail_pct = _trail_percent_for_stage(config, stage)
            if config.submit and state.trail_order_id is not None:
                mod = modify_fn(
                    order_id=int(state.trail_order_id),
                    symbol=sym,
                    trail_percent=trail_pct,
                    account=config.account,
                )
                ok = not mod.get("error") and (
                    mod.get("submitted") or mod.get("status") not in {"rejected"}
                )
                _append_action(
                    state,
                    event="tighten_trail",
                    stage=stage,
                    mid=mid,
                    trail_percent=trail_pct,
                    ok=ok,
                    result=mod,
                )
                if not ok:
                    state.last_error = str(mod.get("error") or "modify failed")
                    state.status = "error"
                    return state
            else:
                _append_action(
                    state,
                    event="tighten_trail_dry",
                    stage=stage,
                    mid=mid,
                    trail_percent=trail_pct,
                )
            state.trail_stage = stage
            state.trail_percent = trail_pct
            state.phase = "trail_active"
            state.last_error = None
            # Last modification done → stop watcher (IB trail remains).
            if stage >= 3:
                state.status = "completed"
                state.phase = "completed"
                _append_action(state, event="final_trail_applied")
            return state

        state.phase = "trail_active"
        return state

    except Exception as exc:
        state.last_error = str(exc)
        state.status = "error"
        _append_action(state, event="error", error=str(exc))
        return state


def run_watcher_loop(job_id: str, config: SellTrailConfig) -> None:
    state = load_state(job_id) or SellTrailState(
        job_id=job_id, config=asdict(config), started_at=_now_iso()
    )
    state.pid = os.getpid()
    state.status = "running"
    state.started_at = state.started_at or _now_iso()
    save_state(state)

    stop = {"flag": False}

    def _handle(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    while not stop["flag"]:
        state = poll_once(config, state)
        state.pid = os.getpid()
        save_state(state)
        if state.status in {"completed", "error"}:
            break
        interval = max(1.0, float(config.interval_sec))
        slept = 0.0
        while slept < interval and not stop["flag"]:
            time.sleep(min(0.5, interval - slept))
            slept += 0.5

    state = load_state(job_id) or state
    if state.status == "running":
        state.status = "stopped"
    state.stopped_at = _now_iso()
    state.pid = os.getpid()
    save_state(state)


def start_background_watcher(
    config: SellTrailConfig, *, job_id: str | None = None
) -> dict:
    import subprocess
    import sys

    jid = job_id or make_job_id(config.normalized_symbol())
    paths = job_paths(jid)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    state = SellTrailState(
        job_id=jid,
        config=asdict(config),
        status="starting",
        phase="waiting_activate",
        started_at=_now_iso(),
    )
    save_state(state)

    cmd = [
        sys.executable,
        "-m",
        "tradingai.ibkr.jobs.sell_trail",
        config.normalized_symbol(),
        "--foreground",
        "--job-id",
        jid,
        "--quantity",
        str(float(config.quantity)),
        "--buy-price",
        str(float(config.buy_price)),
        "--activate-return",
        str(float(config.activate_return)),
        "--initial-trail-percent",
        str(float(config.initial_trail_percent)),
        "--trail-1-5x-percent",
        str(float(config.trail_1_5x_percent)),
        "--trail-2x-percent",
        str(float(config.trail_2x_percent)),
        "--interval",
        str(float(config.interval_sec)),
        "--tif",
        str(config.tif),
    ]
    if config.buy_price_source:
        cmd.extend(["--buy-price-source", str(config.buy_price_source)])
    if config.outside_rth:
        cmd.append("--outside-rth")
    if config.account:
        cmd.extend(["--account", str(config.account)])
    if config.submit:
        cmd.extend(["--submit", "--yes"])

    log_f = open(paths["log"], "a", encoding="utf-8")  # noqa: SIM115
    log_f.write(f"\n--- start {_now_iso()} ---\n")
    log_f.write("cmd: " + " ".join(cmd) + "\n")
    log_f.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(Path.cwd()),
        env=os.environ.copy(),
    )
    state.pid = proc.pid
    state.status = "running"
    save_state(state)
    return {
        "job_id": jid,
        "pid": proc.pid,
        "status": "running",
        "symbol": config.normalized_symbol(),
        "buy_price": config.buy_price,
        "quantity": config.quantity,
        "activate_return": config.activate_return,
        "initial_trail_percent": config.initial_trail_percent,
        "trail_1_5x_percent": config.trail_1_5x_percent,
        "trail_2x_percent": config.trail_2x_percent,
        "interval_sec": config.interval_sec,
        "state_path": str(paths["state"]),
        "log_path": str(paths["log"]),
        "config": asdict(config),
        "message": (
            f"Sell-trail watcher started. Polls every {config.interval_sec:g}s; "
            f"at fee-aware {config.activate_return * 100:g}% profit place "
            f"{config.initial_trail_percent}% trail; tighten to "
            f"{config.trail_1_5x_percent}% @1.5× / "
            f"{config.trail_2x_percent}% @2× activation profit; "
            "exit when sold or final trail applied."
        ),
        "disclaimer": (
            "IBKR sell-trail helper. Real money risk if submit=true. "
            "Not financial advice."
        ),
    }


def resolve_position_and_buy(
    symbol: str,
    *,
    quantity: float | None = None,
    buy_price: float | None = None,
    account: str | None = None,
) -> dict:
    """Load open long + last buy / avg cost from IBKR."""
    from tradingai.intraday.config import get_ibkr_mcp_client_id
    from tradingai.providers.ibkr_tws import (
        fetch_equity_position,
        fetch_last_buy_price,
    )

    ticker = symbol.upper().strip()
    cid = get_ibkr_mcp_client_id() + 55
    position = fetch_equity_position(ticker, account=account, client_id=cid)
    qty_open = float(position.get("quantity") or 0.0)
    if qty_open <= 0:
        raise ValueError(f"No open long position found for {ticker}.")
    qty = float(int(quantity)) if quantity is not None else float(int(qty_open))
    if qty <= 0:
        raise ValueError("quantity must be positive.")
    if qty > qty_open + 1e-9:
        raise ValueError(
            f"quantity {qty} exceeds open position {qty_open} for {ticker}."
        )

    buy_source = None
    price = buy_price
    buy_info = None
    if price is None:
        try:
            buy_info = fetch_last_buy_price(
                ticker, account=account, client_id=cid + 1
            )
        except Exception:
            buy_info = None
        if buy_info and buy_info.get("buy_price"):
            price = float(buy_info["buy_price"])
            buy_source = "last_buy_fill"
        elif position.get("avg_cost"):
            price = float(position["avg_cost"])
            buy_source = "position_avg_cost"
    else:
        buy_source = "explicit"
        price = float(price)
    if price is None or price <= 0:
        raise ValueError(
            f"Could not resolve buy price for {ticker}; pass --buy-price."
        )
    return {
        "symbol": ticker,
        "quantity": qty,
        "position_quantity": qty_open,
        "buy_price": price,
        "buy_price_source": buy_source,
        "position": position,
        "last_buy": buy_info,
    }


def start_sell_trail(
    *,
    symbol: str,
    quantity: float | None = None,
    buy_price: float | None = None,
    activate_return: float = DEFAULT_SELL_TRAIL_ACTIVATE_RETURN,
    initial_trail_percent: float = DEFAULT_SELL_TRAIL_PERCENT,
    trail_1_5x_percent: float = DEFAULT_SELL_TRAIL_1_5X_PERCENT,
    trail_2x_percent: float = DEFAULT_SELL_TRAIL_2X_PERCENT,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
) -> dict:
    """Resolve buy/position; dry-run plan or start background sell-trail watcher."""
    meta = resolve_position_and_buy(
        symbol, quantity=quantity, buy_price=buy_price, account=account
    )
    qty = float(meta["quantity"])
    buy_px = float(meta["buy_price"])
    act = float(activate_return)
    activate_mid = _compute_profit_mid(
        buy_price=buy_px, shares=qty, net_return=act
    )
    mid_1_5x = _compute_profit_mid(
        buy_price=buy_px, shares=qty, net_return=act * 1.5
    )
    mid_2x = _compute_profit_mid(
        buy_price=buy_px, shares=qty, net_return=act * 2.0
    )
    plan = {
        "symbol": meta["symbol"],
        "quantity": qty,
        "buy_price": buy_px,
        "buy_price_source": meta["buy_price_source"],
        "activate_return": act,
        "activate_mid": activate_mid,
        "mid_1_5x": mid_1_5x,
        "mid_2x": mid_2x,
        "initial_trail_percent": initial_trail_percent,
        "trail_1_5x_percent": trail_1_5x_percent,
        "trail_2x_percent": trail_2x_percent,
        "interval_sec": interval_sec,
        "note": (
            f"Wait until mid ≥ fee-aware {act * 100:g}% profit → "
            f"TRAIL {initial_trail_percent}%; @1.5× → {trail_1_5x_percent}%; "
            f"@2× → {trail_2x_percent}% then exit watcher; also exit if flat."
        ),
    }

    if not submit:
        return {
            "position": meta["position"],
            "last_buy": meta["last_buy"],
            "watcher": None,
            "plan": plan,
            "message": (
                "Dry-run plan only — no watcher. "
                "Re-run with --submit to start monitoring."
            ),
            "disclaimer": (
                "IBKR sell-trail helper. Real money risk if submit=true. "
                "Not financial advice."
            ),
        }

    config = SellTrailConfig(
        symbol=meta["symbol"],
        quantity=qty,
        buy_price=buy_px,
        activate_return=act,
        initial_trail_percent=float(initial_trail_percent),
        trail_1_5x_percent=float(trail_1_5x_percent),
        trail_2x_percent=float(trail_2x_percent),
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        submit=True,
        interval_sec=float(interval_sec),
        buy_price_source=meta["buy_price_source"],
    )
    watcher = start_background_watcher(config)
    return {
        "position": meta["position"],
        "last_buy": meta["last_buy"],
        "plan": plan,
        "watcher": watcher,
        "message": watcher.get("message"),
        "disclaimer": watcher.get("disclaimer"),
    }
