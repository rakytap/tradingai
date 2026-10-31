"""Budget LIMIT buy + stepped trail: 0.6% → 0.05%/0.03%/0.01% at 1×/1.5×/2× target."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingai.intraday.config import get_target_return
from tradingai.intraday.profit_target import fee_aware_sell_price
from tradingai.providers.ibkr_tws import (
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_BUY_TRAIL_1_5X_PERCENT,
    DEFAULT_BUY_TRAIL_2X_PERCENT,
    DEFAULT_BUY_TRAIL_PERCENT,
    DEFAULT_BUY_TRAIL_TIGHT_PERCENT,
)

DEFAULT_POLL_INTERVAL_SEC = 10.0
# After first tighten (0.05%), poll faster.
DEFAULT_TIGHT_POLL_INTERVAL_SEC = 5.0


def buy_trail_state_dir() -> Path:
    raw = (os.getenv("TRADINGAI_BUY_TRAIL_DIR") or "").strip()
    if raw:
        path = Path(raw).expanduser()
    else:
        path = Path("data") / "buy_trail"
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
class BuyTrailConfig:
    symbol: str
    buy_order_id: int
    target_qty: float
    activate_price: float
    # Trail params (IB percent points).
    initial_trail_percent: float = DEFAULT_BUY_TRAIL_PERCENT
    tight_trail_percent: float = DEFAULT_BUY_TRAIL_TIGHT_PERCENT
    trail_1_5x_percent: float = DEFAULT_BUY_TRAIL_1_5X_PERCENT
    trail_2x_percent: float = DEFAULT_BUY_TRAIL_2X_PERCENT
    target_return: float | None = None
    tif: str = "DAY"
    outside_rth: bool = False
    account: str | None = None
    submit: bool = False
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
    tight_interval_sec: float = DEFAULT_TIGHT_POLL_INTERVAL_SEC
    total_budget: float | None = None
    budget_fraction: float = DEFAULT_BUDGET_FRACTION
    buy_limit_price: float | None = None

    def normalized_symbol(self) -> str:
        return self.symbol.upper().strip()

    def resolved_target_return(self) -> float:
        return (
            float(self.target_return)
            if self.target_return is not None
            else float(get_target_return())
        )


@dataclass
class BuyTrailState:
    job_id: str
    config: dict
    pid: int | None = None
    # waiting_fill | trail_active | trail_tight | completed | stopped | error
    status: str = "starting"
    phase: str = "waiting_fill"
    started_at: str | None = None
    stopped_at: str | None = None
    last_poll_at: str | None = None
    last_error: str | None = None
    polls: int = 0
    filled_shares: float = 0.0
    avg_fill_price: float | None = None
    buy_complete: bool = False
    trail_order_id: int | None = None
    trail_percent: float | None = None
    # 0=0.6% initial, 1=0.05% @1×, 2=0.03% @1.5×, 3=0.01% @2×
    trail_stage: int = 0
    target_mid: float | None = None
    target_mid_1_5x: float | None = None
    target_mid_2x: float | None = None
    last_mid: float | None = None
    tightened: bool = False
    actions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BuyTrailState:
        return cls(
            job_id=str(data["job_id"]),
            config=dict(data.get("config") or {}),
            pid=data.get("pid"),
            status=str(data.get("status") or "stopped"),
            phase=str(data.get("phase") or "waiting_fill"),
            started_at=data.get("started_at"),
            stopped_at=data.get("stopped_at"),
            last_poll_at=data.get("last_poll_at"),
            last_error=data.get("last_error"),
            polls=int(data.get("polls") or 0),
            filled_shares=float(data.get("filled_shares") or 0.0),
            avg_fill_price=(
                float(data["avg_fill_price"])
                if data.get("avg_fill_price") is not None
                else None
            ),
            buy_complete=bool(data.get("buy_complete")),
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
            target_mid=(
                float(data["target_mid"]) if data.get("target_mid") is not None else None
            ),
            target_mid_1_5x=(
                float(data["target_mid_1_5x"])
                if data.get("target_mid_1_5x") is not None
                else None
            ),
            target_mid_2x=(
                float(data["target_mid_2x"])
                if data.get("target_mid_2x") is not None
                else None
            ),
            last_mid=(
                float(data["last_mid"]) if data.get("last_mid") is not None else None
            ),
            tightened=bool(data.get("tightened")),
            actions=list(data.get("actions") or []),
        )


def job_paths(job_id: str) -> dict[str, Path]:
    base = buy_trail_state_dir() / job_id
    return {
        "dir": base,
        "state": base / "state.json",
        "pid": base / "job.pid",
        "log": base / "job.log",
    }


def save_state(state: BuyTrailState) -> Path:
    paths = job_paths(state.job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["state"].write_text(
        json.dumps(state.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    if state.pid:
        paths["pid"].write_text(str(state.pid) + "\n", encoding="utf-8")
    return paths["state"]


def load_state(job_id: str) -> BuyTrailState | None:
    path = job_paths(job_id)["state"]
    if not path.is_file():
        return None
    return BuyTrailState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def make_job_id(symbol: str, buy_order_id: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol.upper().strip()}_{int(buy_order_id)}_{stamp}"


def list_jobs() -> list[dict]:
    root = buy_trail_state_dir()
    out: list[dict] = []
    for path in sorted(root.glob("*/state.json")):
        try:
            state = BuyTrailState.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
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
                "buy_order_id": cfg.get("buy_order_id"),
                "filled_shares": state.filled_shares,
                "trail_order_id": state.trail_order_id,
                "trail_percent": state.trail_percent,
                "trail_stage": state.trail_stage,
                "tightened": state.tightened,
                "buy_complete": state.buy_complete,
                "started_at": state.started_at,
                "log": str(job_paths(state.job_id)["log"]),
            }
        )
    return out


def job_status(job_id: str | None = None) -> dict:
    if job_id:
        state = load_state(job_id)
        if state is None:
            return {"error": f"Unknown job_id {job_id!r}", "job_id": job_id}
        alive = _pid_alive(state.pid)
        return {
            "job_id": job_id,
            "alive": alive,
            "status": state.status if alive or state.status != "running" else "dead",
            "phase": state.phase,
            "pid": state.pid,
            "config": state.config,
            "started_at": state.started_at,
            "stopped_at": state.stopped_at,
            "last_poll_at": state.last_poll_at,
            "last_error": state.last_error,
            "polls": state.polls,
            "filled_shares": state.filled_shares,
            "avg_fill_price": state.avg_fill_price,
            "buy_complete": state.buy_complete,
            "trail_order_id": state.trail_order_id,
            "trail_percent": state.trail_percent,
            "trail_stage": state.trail_stage,
            "target_mid": state.target_mid,
            "target_mid_1_5x": state.target_mid_1_5x,
            "target_mid_2x": state.target_mid_2x,
            "last_mid": state.last_mid,
            "tightened": state.tightened,
            "actions": state.actions[-20:],
            "log": str(job_paths(job_id)["log"]),
        }
    return {"jobs": list_jobs(), "state_dir": str(buy_trail_state_dir())}


def stop_job(job_id: str, force: bool = False) -> dict:
    state = load_state(job_id)
    if state is None:
        return {"error": f"Unknown job_id {job_id!r}", "job_id": job_id}
    pid = state.pid
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if force:
            time.sleep(0.5)
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    state.status = "stopped"
    state.stopped_at = _now_iso()
    save_state(state)
    return {
        "job_id": job_id,
        "status": "stopped",
        "pid": pid,
        "alive": _pid_alive(pid),
    }


def _append_action(state: BuyTrailState, **kwargs: Any) -> None:
    state.actions.append({"at": _now_iso(), **kwargs})
    state.actions = state.actions[-100:]


def _trail_stop_from_activate(activate: float, trail_percent: float) -> float:
    """Initial protective stop: activate_mid * (1 - trail%/100)."""
    from tradingai.providers.ibkr_tws import round_us_equity_price

    px = float(activate) * (1.0 - float(trail_percent) / 100.0)
    return round_us_equity_price(px, mode="floor")


def _compute_target_mid(
    *,
    buy_price: float,
    shares: float,
    target_return: float,
) -> float:
    from tradingai.providers.ibkr_tws import round_us_equity_price

    meta = fee_aware_sell_price(
        buy_price,
        buy_notional_usd=buy_price * shares,
        net_return=target_return,
    )
    return round_us_equity_price(float(meta["sell_price"]), mode="ceil")


def _desired_trail_stage(mid: float, state: BuyTrailState) -> int:
    """Return highest stage unlocked by mid vs 1× / 1.5× / 2× target mids."""
    stage = 0
    if state.target_mid is not None and mid + 1e-9 >= float(state.target_mid):
        stage = 1
    if state.target_mid_1_5x is not None and mid + 1e-9 >= float(state.target_mid_1_5x):
        stage = 2
    if state.target_mid_2x is not None and mid + 1e-9 >= float(state.target_mid_2x):
        stage = 3
    return stage


def _trail_percent_for_stage(config: BuyTrailConfig, stage: int) -> float:
    if stage >= 3:
        return float(config.trail_2x_percent)
    if stage == 2:
        return float(config.trail_1_5x_percent)
    if stage == 1:
        return float(config.tight_trail_percent)
    return float(config.initial_trail_percent)


def _apply_trail_stage(
    *,
    config: BuyTrailConfig,
    state: BuyTrailState,
    mid: float,
    stage: int,
    modify_fn,
    sym: str,
) -> BuyTrailState:
    """Modify working trail to the percent for ``stage`` (1/2/3)."""
    trail_pct = _trail_percent_for_stage(config, stage)
    new_stop = _trail_stop_from_activate(mid, trail_pct)
    if config.submit and state.trail_order_id is not None:
        mod = modify_fn(
            order_id=int(state.trail_order_id),
            trail_percent=trail_pct,
            trail_stop_price=new_stop,
            symbol=sym,
        )
        ok = bool(mod.get("submitted")) and not mod.get("error")
        _append_action(
            state,
            event="tighten_trail",
            stage=stage,
            mid=mid,
            target_mid=state.target_mid,
            target_mid_1_5x=state.target_mid_1_5x,
            target_mid_2x=state.target_mid_2x,
            trail_percent=trail_pct,
            trail_stop_price=new_stop,
            ok=ok,
            status=mod.get("status"),
            error=mod.get("error"),
        )
        if ok:
            state.trail_stage = stage
            state.tightened = stage >= 1
            state.trail_percent = trail_pct
            state.phase = "trail_tight"
            state.last_error = None
        else:
            state.last_error = str(mod.get("error") or mod.get("status"))
    else:
        state.trail_stage = stage
        state.tightened = stage >= 1
        state.trail_percent = trail_pct
        state.phase = "trail_tight"
        _append_action(
            state,
            event="tighten_trail_dry",
            stage=stage,
            mid=mid,
            trail_percent=trail_pct,
        )
    return state


def _position_flat(position_fn, *, symbol: str, account: str | None) -> bool | None:
    """True if flat, False if still long, None if position lookup failed."""
    try:
        pos = position_fn(symbol=symbol, account=account)
        return float(pos.get("quantity") or 0.0) <= 0.0
    except Exception:
        return None


def poll_once(
    config: BuyTrailConfig,
    state: BuyTrailState,
    *,
    snapshot_fn=None,
    mid_fn=None,
    sell_fn=None,
    modify_fn=None,
    position_fn=None,
) -> BuyTrailState:
    """Advance fill → place 0.6% trail → tighten to 0.05% at target mid."""
    from tradingai.ibkr.tools.orders import sell_stock_trail_tool
    from tradingai.intraday.config import get_ibkr_mcp_client_id
    from tradingai.providers.ibkr_tws import (
        fetch_equity_mid_price,
        fetch_equity_position,
        fetch_order_snapshot,
        modify_trail_limit_percent,
    )

    cid = get_ibkr_mcp_client_id() + 40
    if snapshot_fn is None:

        def snapshot_fn(**kwargs):  # type: ignore[misc]
            return fetch_order_snapshot(**kwargs, client_id=cid)

    if mid_fn is None:

        def mid_fn(**kwargs):  # type: ignore[misc]
            return fetch_equity_mid_price(**kwargs, client_id=cid + 1)

    if sell_fn is None:
        sell_fn = sell_stock_trail_tool

    if modify_fn is None:

        def modify_fn(**kwargs):  # type: ignore[misc]
            return modify_trail_limit_percent(**kwargs, client_id=cid + 2)

    if position_fn is None:

        def position_fn(**kwargs):  # type: ignore[misc]
            return fetch_equity_position(**kwargs, client_id=cid + 3)

    state.polls += 1
    state.last_poll_at = _now_iso()
    state.status = "running"
    sym = config.normalized_symbol()

    try:
        snap = snapshot_fn(order_id=int(config.buy_order_id), symbol=sym)
        if snap.get("error"):
            raise RuntimeError(str(snap["error"]))

        open_order = snap.get("open_order") or {}
        filled = float(snap.get("filled_shares") or 0.0)
        open_filled = float(open_order.get("filled") or 0.0)
        state.filled_shares = max(filled, open_filled)
        avg = open_order.get("avg_fill_price")
        if avg:
            state.avg_fill_price = float(avg)
        elif snap.get("fills"):
            # VWAP from fills
            num = sum(
                float(f.get("shares") or 0) * float(f.get("price") or 0)
                for f in snap["fills"]
            )
            den = sum(float(f.get("shares") or 0) for f in snap["fills"])
            if den > 0:
                state.avg_fill_price = num / den

        still_open = bool(snap.get("open"))
        state.buy_complete = (not still_open) and state.filled_shares > 0
        buy_gone_empty = (not still_open) and state.filled_shares <= 0

        if buy_gone_empty and state.trail_order_id is None:
            state.phase = "completed"
            state.status = "completed"
            state.last_error = None
            _append_action(state, event="buy_done_unfilled")
            return state

        # Phase 1: wait for full fill.
        if not state.buy_complete:
            state.phase = "waiting_fill"
            return state

        qty = float(int(state.filled_shares))
        if qty < 1:
            state.phase = "completed"
            state.status = "completed"
            return state

        buy_px = float(
            state.avg_fill_price
            or config.buy_limit_price
            or config.activate_price
        )
        ret = config.resolved_target_return()
        if state.target_mid is None:
            state.target_mid = _compute_target_mid(
                buy_price=buy_px, shares=qty, target_return=ret
            )
        if state.target_mid_1_5x is None:
            state.target_mid_1_5x = _compute_target_mid(
                buy_price=buy_px, shares=qty, target_return=ret * 1.5
            )
        if state.target_mid_2x is None:
            state.target_mid_2x = _compute_target_mid(
                buy_price=buy_px, shares=qty, target_return=ret * 2.0
            )

        # Phase 2: place initial trail once filled.
        if state.trail_order_id is None:
            stop_px = _trail_stop_from_activate(
                float(config.activate_price),
                float(config.initial_trail_percent),
            )
            result = sell_fn(
                symbol=sym,
                quantity=qty,
                limit_price=stop_px,
                trail_percent=float(config.initial_trail_percent),
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
                trail_percent=config.initial_trail_percent,
                trail_stop_price=stop_px,
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
                state.trail_percent = float(config.initial_trail_percent)
                state.trail_stage = 0
                state.phase = "trail_active"
                state.last_error = None
                if not config.submit:
                    # Dry-run watcher exits after simulating trail place.
                    state.status = "completed"
                    state.phase = "completed"
            else:
                state.last_error = str(
                    result.get("error") or result.get("message") or "trail failed"
                )
            return state

        # Exit as soon as the long is gone (trail filled / position sold).
        flat = _position_flat(position_fn, symbol=sym, account=config.account)
        if flat is True:
            state.phase = "completed"
            state.status = "completed"
            state.last_error = None
            _append_action(state, event="position_depleted")
            return state

        # Phase 3: poll mid; step trail 0.05% → 0.03% → 0.01% at 1×/1.5×/2×.
        mid_info = mid_fn(symbol=sym)
        mid = float(mid_info["mid_price"])
        state.last_mid = mid
        desired = _desired_trail_stage(mid, state)
        if desired > int(state.trail_stage):
            state = _apply_trail_stage(
                config=config,
                state=state,
                mid=mid,
                stage=desired,
                modify_fn=modify_fn,
                sym=sym,
            )
            if not config.submit and state.trail_stage >= 1:
                # Dry-run: stop after first simulated tighten.
                state.status = "completed"
            return state

        state.phase = "trail_tight" if state.trail_stage >= 1 else "trail_active"
        return state

    except Exception as exc:
        state.last_error = str(exc)
        state.status = "error"
        _append_action(state, event="error", error=str(exc))
        return state


def run_watcher_loop(job_id: str, config: BuyTrailConfig) -> None:
    state = load_state(job_id) or BuyTrailState(
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
        # After first tighten (0.05%), poll every 5s; otherwise 10s.
        if int(state.trail_stage) >= 1:
            interval = max(1.0, float(config.tight_interval_sec))
        else:
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


def start_background_watcher(config: BuyTrailConfig, *, job_id: str | None = None) -> dict:
    import subprocess
    import sys

    jid = job_id or make_job_id(config.normalized_symbol(), config.buy_order_id)
    paths = job_paths(jid)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    state = BuyTrailState(
        job_id=jid,
        config=asdict(config),
        status="starting",
        phase="waiting_fill",
        started_at=_now_iso(),
    )
    save_state(state)

    cmd = [
        sys.executable,
        "-m",
        "tradingai.ibkr.jobs.buy_trail",
        config.normalized_symbol(),
        "--foreground",
        "--job-id",
        jid,
        "--buy-order-id",
        str(int(config.buy_order_id)),
        "--target-qty",
        str(float(config.target_qty)),
        "--activate-price",
        str(float(config.activate_price)),
        "--interval",
        str(config.interval_sec),
        "--tight-interval",
        str(config.tight_interval_sec),
        "--initial-trail-percent",
        str(config.initial_trail_percent),
        "--tight-trail-percent",
        str(config.tight_trail_percent),
        "--trail-1-5x-percent",
        str(config.trail_1_5x_percent),
        "--trail-2x-percent",
        str(config.trail_2x_percent),
        "--tif",
        str(config.tif),
    ]
    if config.target_return is not None:
        cmd.extend(["--target-return", str(config.target_return)])
    if config.buy_limit_price is not None:
        cmd.extend(["--buy-limit-price", str(config.buy_limit_price)])
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
        "buy_order_id": config.buy_order_id,
        "activate_price": config.activate_price,
        "initial_trail_percent": config.initial_trail_percent,
        "tight_trail_percent": config.tight_trail_percent,
        "trail_1_5x_percent": config.trail_1_5x_percent,
        "trail_2x_percent": config.trail_2x_percent,
        "interval_sec": config.interval_sec,
        "tight_interval_sec": config.tight_interval_sec,
        "state_path": str(paths["state"]),
        "log_path": str(paths["log"]),
        "config": asdict(config),
        "message": (
            "Buy-trail watcher started. Polls every "
            f"{config.interval_sec:g}s until 1× target (then "
            f"{config.tight_interval_sec:g}s): after fill place "
            f"{config.initial_trail_percent}% trail; tighten to "
            f"{config.tight_trail_percent}% @1×, "
            f"{config.trail_1_5x_percent}% @1.5×, "
            f"{config.trail_2x_percent}% @2× fee-aware target; "
            "exit when position is flat."
        ),
        "disclaimer": (
            "IBKR buy-trail helper. Real money risk if submit=true. "
            "Not financial advice."
        ),
    }


def start_buy_trail(
    *,
    symbol: str,
    total_budget: float | None = None,
    limit_price: float | None = None,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
    initial_trail_percent: float = DEFAULT_BUY_TRAIL_PERCENT,
    tight_trail_percent: float = DEFAULT_BUY_TRAIL_TIGHT_PERCENT,
    trail_1_5x_percent: float = DEFAULT_BUY_TRAIL_1_5X_PERCENT,
    trail_2x_percent: float = DEFAULT_BUY_TRAIL_2X_PERCENT,
    target_return: float | None = None,
    tif: str = "DAY",
    outside_rth: bool = False,
    account: str | None = None,
    submit: bool = False,
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    tight_interval_sec: float = DEFAULT_TIGHT_POLL_INTERVAL_SEC,
) -> dict:
    """LIMIT buy (no STP); on live submit, watch fill → trail → tighten."""
    from tradingai.ibkr.tools.orders import buy_stock_budget_tool

    buy = buy_stock_budget_tool(
        symbol=symbol,
        total_budget=total_budget,
        limit_price=limit_price,
        budget_fraction=budget_fraction,
        stop_loss=None,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        dry_run=not bool(submit),
    )
    if buy.get("error") or buy.get("status") == "rejected":
        return {
            "error": buy.get("error") or buy.get("message") or "buy rejected",
            "buy": buy,
            "watcher": None,
        }

    sizing = buy.get("sizing") or buy
    target_qty = float(sizing.get("quantity") or buy.get("quantity") or 0)
    activate = float(
        sizing.get("limit_price")
        or buy.get("limit_price")
        or limit_price
        or 0
    )
    if activate <= 0:
        return {
            "error": "Could not resolve buy mid / limit as trail activate price.",
            "buy": buy,
            "watcher": None,
        }

    order_id = buy.get("order_id")
    if submit and not buy.get("submitted"):
        return {
            "error": "Buy was not submitted; watcher not started.",
            "buy": buy,
            "watcher": None,
        }
    if submit and order_id is None:
        return {
            "error": "Buy submitted but no order_id; watcher not started.",
            "buy": buy,
            "watcher": None,
        }

    if not submit:
        ret = (
            float(target_return)
            if target_return is not None
            else float(get_target_return())
        )
        preview_stop = _trail_stop_from_activate(activate, initial_trail_percent)
        return {
            "buy": buy,
            "watcher": None,
            "plan": {
                "activate_price": activate,
                "initial_trail_percent": initial_trail_percent,
                "initial_trail_stop_price": preview_stop,
                "tight_trail_percent": tight_trail_percent,
                "trail_1_5x_percent": trail_1_5x_percent,
                "trail_2x_percent": trail_2x_percent,
                "target_return": ret,
                "interval_sec": interval_sec,
                "tight_interval_sec": tight_interval_sec,
                "note": (
                    "After fill: TRAIL LIMIT at stop=activate*(1-0.6/100); "
                    "1× target → 0.05% (poll 5s); 1.5× → 0.03%; 2× → 0.01%; "
                    "exit when position flat."
                ),
            },
            "message": (
                "Dry-run buy only — no background watcher. "
                "Re-run with submit=true to buy and start trail monitoring."
            ),
            "disclaimer": (
                "IBKR buy-trail helper. Real money risk if submit=true. "
                "Not financial advice."
            ),
        }

    config = BuyTrailConfig(
        symbol=symbol,
        buy_order_id=int(order_id),
        target_qty=target_qty,
        activate_price=activate,
        initial_trail_percent=float(initial_trail_percent),
        tight_trail_percent=float(tight_trail_percent),
        trail_1_5x_percent=float(trail_1_5x_percent),
        trail_2x_percent=float(trail_2x_percent),
        target_return=target_return,
        tif=tif,
        outside_rth=outside_rth,
        account=account,
        submit=True,
        interval_sec=float(interval_sec),
        tight_interval_sec=float(tight_interval_sec),
        total_budget=total_budget,
        budget_fraction=budget_fraction,
        buy_limit_price=activate,
    )
    watcher = start_background_watcher(config)
    return {
        "buy": buy,
        "watcher": watcher,
        "message": watcher.get("message"),
        "disclaimer": watcher.get("disclaimer"),
    }
