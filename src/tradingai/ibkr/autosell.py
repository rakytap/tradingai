"""Background autosell: pair IBKR buys with sells and trail-sell unpaired lots."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingai.providers.ibkr_tws import DEFAULT_SELL_TRAIL_PERCENT

DEFAULT_POLL_INTERVAL_SEC = 15.0
DEFAULT_TRADE_LIMIT = 200


def autosell_state_dir() -> Path:
    raw = (os.getenv("TRADINGAI_AUTOSELL_DIR") or "").strip()
    if raw:
        path = Path(raw).expanduser()
    else:
        path = Path("data") / "autosell"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trade_key(trade: dict) -> str:
    exec_id = str(trade.get("exec_id") or "").strip()
    if exec_id:
        return exec_id
    return "|".join(
        [
            str(trade.get("time") or ""),
            str(trade.get("symbol") or ""),
            str(trade.get("side") or ""),
            str(trade.get("shares") or ""),
            str(trade.get("price") or ""),
            str(trade.get("order_id") or ""),
        ]
    )


def find_unpaired_buys(trades: list[dict]) -> list[dict]:
    """FIFO-match sells against buys; return remaining buy lots (oldest first).

    Each item includes ``shares`` left unpaired and the original fill fields.
    """
    chronological = sorted(
        trades,
        key=lambda t: (t.get("time") or "", _trade_key(t)),
    )
    open_lots: list[dict] = []
    for trade in chronological:
        side = (trade.get("side") or "").lower()
        shares = float(trade.get("shares") or 0.0)
        if shares <= 0:
            continue
        if side == "buy":
            lot = dict(trade)
            lot["shares"] = shares
            lot["original_shares"] = shares
            lot["buy_key"] = _trade_key(trade)
            open_lots.append(lot)
            continue
        if side != "sell":
            continue
        remaining = shares
        while remaining > 1e-9 and open_lots:
            lot = open_lots[0]
            take = min(float(lot["shares"]), remaining)
            lot["shares"] = float(lot["shares"]) - take
            remaining -= take
            if lot["shares"] <= 1e-9:
                open_lots.pop(0)
    unpaired: list[dict] = []
    for lot in open_lots:
        left = float(lot["shares"])
        if left <= 1e-9:
            continue
        out = dict(lot)
        out["shares"] = round(left, 8) if abs(left - round(left)) < 1e-8 else left
        unpaired.append(out)
    return unpaired


@dataclass
class AutosellConfig:
    symbol: str
    quantity: float | None = None  # optional per-sell cap
    limit_price: float | None = None
    target_return: float | None = None
    trail_percent: float = DEFAULT_SELL_TRAIL_PERCENT
    tif: str = "DAY"
    outside_rth: bool = False
    account: str | None = None
    submit: bool = False
    interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
    trade_limit: int = DEFAULT_TRADE_LIMIT

    def normalized_symbol(self) -> str:
        return self.symbol.upper().strip()


@dataclass
class AutosellState:
    job_id: str
    config: dict
    pid: int | None = None
    status: str = "starting"  # starting|running|stopped|error
    started_at: str | None = None
    stopped_at: str | None = None
    last_poll_at: str | None = None
    last_error: str | None = None
    handled_buy_keys: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    polls: int = 0
    sells_attempted: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AutosellState:
        return cls(
            job_id=str(data["job_id"]),
            config=dict(data.get("config") or {}),
            pid=data.get("pid"),
            status=str(data.get("status") or "stopped"),
            started_at=data.get("started_at"),
            stopped_at=data.get("stopped_at"),
            last_poll_at=data.get("last_poll_at"),
            last_error=data.get("last_error"),
            handled_buy_keys=list(data.get("handled_buy_keys") or []),
            actions=list(data.get("actions") or []),
            polls=int(data.get("polls") or 0),
            sells_attempted=int(data.get("sells_attempted") or 0),
        )


def job_paths(job_id: str) -> dict[str, Path]:
    base = autosell_state_dir() / job_id
    return {
        "dir": base,
        "state": base / "state.json",
        "pid": base / "job.pid",
        "log": base / "job.log",
    }


def save_state(state: AutosellState) -> Path:
    paths = job_paths(state.job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["state"].write_text(
        json.dumps(state.to_dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if state.pid is not None:
        paths["pid"].write_text(f"{state.pid}\n", encoding="utf-8")
    return paths["state"]


def load_state(job_id: str) -> AutosellState | None:
    path = job_paths(job_id)["state"]
    if not path.is_file():
        return None
    return AutosellState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_jobs() -> list[dict]:
    root = autosell_state_dir()
    jobs: list[dict] = []
    if not root.is_dir():
        return jobs
    for child in sorted(root.iterdir()):
        state_path = child / "state.json"
        if not state_path.is_file():
            continue
        try:
            state = AutosellState.from_dict(
                json.loads(state_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            jobs.append({"job_id": child.name, "error": str(exc)})
            continue
        alive = _pid_alive(state.pid)
        jobs.append(
            {
                "job_id": state.job_id,
                "status": state.status if alive or state.status != "running" else "dead",
                "pid": state.pid,
                "alive": alive,
                "symbol": (state.config or {}).get("symbol"),
                "submit": bool((state.config or {}).get("submit")),
                "started_at": state.started_at,
                "last_poll_at": state.last_poll_at,
                "sells_attempted": state.sells_attempted,
                "log": str(job_paths(state.job_id)["log"]),
            }
        )
    return jobs


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


def stop_job(job_id: str, *, force: bool = False) -> dict:
    state = load_state(job_id)
    if state is None:
        return {"error": f"Unknown job_id {job_id!r}", "job_id": job_id}
    pid = state.pid
    stopped = False
    if _pid_alive(pid):
        assert pid is not None
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.kill(pid, sig)
        for _ in range(20):
            if not _pid_alive(pid):
                stopped = True
                break
            time.sleep(0.1)
        if not stopped and not force:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.2)
            stopped = not _pid_alive(pid)
    else:
        stopped = True
    state.status = "stopped"
    state.stopped_at = _now_iso()
    state.last_error = None
    save_state(state)
    return {
        "job_id": job_id,
        "stopped": stopped,
        "pid": pid,
        "status": state.status,
        "state_path": str(job_paths(job_id)["state"]),
    }


def make_job_id(symbol: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol.upper().strip()}_{stamp}"


def _sell_qty_for_lot(lot: dict, config: AutosellConfig) -> float:
    qty = float(lot.get("shares") or 0.0)
    if config.quantity is not None and float(config.quantity) > 0:
        qty = min(qty, float(config.quantity))
    # Whole shares for US equities trail sells.
    return float(int(qty))


def poll_once(
    config: AutosellConfig,
    state: AutosellState,
    *,
    fetch_trades_fn=None,
    sell_fn=None,
) -> AutosellState:
    """One poll: find unpaired buys and trail-sell new ones."""
    from tradingai.ibkr.tools.orders import sell_stock_trail_tool
    from tradingai.intraday.config import get_ibkr_mcp_client_id
    from tradingai.providers.ibkr_tws import fetch_recent_trades

    if fetch_trades_fn is None:
        def fetch_trades_fn(**kwargs):  # type: ignore[misc]
            return fetch_recent_trades(
                **kwargs,
                client_id=get_ibkr_mcp_client_id() + 20,
            )

    if sell_fn is None:
        sell_fn = sell_stock_trail_tool

    state.polls += 1
    state.last_poll_at = _now_iso()
    state.status = "running"
    handled = set(state.handled_buy_keys)

    try:
        raw = fetch_trades_fn(
            limit=int(config.trade_limit),
            side="all",
            symbol=config.normalized_symbol(),
            account=config.account,
        )
        if raw.get("error"):
            raise RuntimeError(str(raw["error"]))
        trades = list(raw.get("trades") or [])
        unpaired = find_unpaired_buys(trades)
        for lot in unpaired:
            buy_key = str(lot.get("buy_key") or _trade_key(lot))
            if buy_key in handled:
                continue
            qty = _sell_qty_for_lot(lot, config)
            if qty < 1:
                action = {
                    "at": _now_iso(),
                    "buy_key": buy_key,
                    "skipped": True,
                    "reason": "quantity < 1 after floor/cap",
                    "lot_shares": lot.get("shares"),
                }
                state.actions.append(action)
                handled.add(buy_key)
                continue

            result = sell_fn(
                symbol=config.normalized_symbol(),
                quantity=qty,
                limit_price=config.limit_price,
                trail_percent=config.trail_percent,
                target_return=config.target_return,
                tif=config.tif,
                outside_rth=config.outside_rth,
                account=config.account,
                dry_run=not bool(config.submit),
            )
            state.sells_attempted += 1
            ok = not (
                result.get("error")
                or result.get("status") in {
                    "rejected",
                    "Inactive",
                    "Cancelled",
                    "ApiCancelled",
                    "Rejected",
                }
            )
            # Mark handled on successful submit or successful dry-run preview.
            if ok and (result.get("submitted") or not config.submit):
                handled.add(buy_key)
            action = {
                "at": _now_iso(),
                "buy_key": buy_key,
                "buy_time": lot.get("time"),
                "buy_price": lot.get("price"),
                "quantity": qty,
                "submit": bool(config.submit),
                "ok": ok,
                "result_status": result.get("status"),
                "order_id": result.get("order_id"),
                "error": result.get("error"),
                "message": result.get("message"),
            }
            state.actions.append(action)
            # Keep action log bounded.
            state.actions = state.actions[-100:]
            if not ok:
                state.last_error = str(result.get("error") or result.get("message") or "sell failed")
            else:
                state.last_error = None
        state.handled_buy_keys = sorted(handled)
    except Exception as exc:
        state.last_error = str(exc)
    return state


def run_watcher_loop(job_id: str, config: AutosellConfig) -> None:
    """Foreground poll loop until SIGTERM/SIGINT."""
    state = load_state(job_id) or AutosellState(
        job_id=job_id,
        config=asdict(config),
        started_at=_now_iso(),
    )
    state.config = asdict(config)
    state.pid = os.getpid()
    state.status = "running"
    state.started_at = state.started_at or _now_iso()
    save_state(state)

    stop = {"flag": False}

    def _handle_stop(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    interval = max(1.0, float(config.interval_sec))
    while not stop["flag"]:
        state = load_state(job_id) or state
        # Preserve runtime pid/status across reloads.
        state.pid = os.getpid()
        state = poll_once(config, state)
        save_state(state)
        # Sleep in chunks so SIGTERM is responsive.
        slept = 0.0
        while slept < interval and not stop["flag"]:
            time.sleep(min(0.5, interval - slept))
            slept += 0.5

    state = load_state(job_id) or state
    state.status = "stopped"
    state.stopped_at = _now_iso()
    state.pid = os.getpid()
    save_state(state)


def start_background_job(config: AutosellConfig, *, job_id: str | None = None) -> dict:
    """Spawn ``tradingai-autosell --foreground`` as a detached process."""
    import subprocess
    import sys

    jid = job_id or make_job_id(config.normalized_symbol())
    paths = job_paths(jid)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    state = AutosellState(
        job_id=jid,
        config=asdict(config),
        status="starting",
        started_at=_now_iso(),
    )
    save_state(state)

    cmd = [
        sys.executable,
        "-m",
        "tradingai.ibkr.jobs.autosell_watch",
        config.normalized_symbol(),
        "--foreground",
        "--job-id",
        jid,
        "--interval",
        str(config.interval_sec),
        "--trail-percent",
        str(config.trail_percent),
        "--tif",
        str(config.tif),
    ]
    if config.quantity is not None:
        cmd.extend(["--qty", str(config.quantity)])
    if config.limit_price is not None:
        cmd.extend(["--price", str(config.limit_price)])
    if config.target_return is not None:
        cmd.extend(["--target-return", str(config.target_return)])
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
        "submit": bool(config.submit),
        "interval_sec": config.interval_sec,
        "state_path": str(paths["state"]),
        "log_path": str(paths["log"]),
        "config": asdict(config),
        "message": (
            "Autosell watcher started in background. "
            "Polls IBKR fills every "
            f"{config.interval_sec:g}s and trail-sells unpaired buys."
        ),
        "disclaimer": (
            "IBKR autosell helper. Real money risk if submit=true. "
            "Not financial advice."
        ),
    }


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
            "pid": state.pid,
            "config": state.config,
            "started_at": state.started_at,
            "stopped_at": state.stopped_at,
            "last_poll_at": state.last_poll_at,
            "last_error": state.last_error,
            "polls": state.polls,
            "sells_attempted": state.sells_attempted,
            "handled_buy_keys": state.handled_buy_keys,
            "actions": state.actions[-20:],
            "log_path": str(job_paths(job_id)["log"]),
            "state_path": str(job_paths(job_id)["state"]),
        }
    return {"jobs": list_jobs(), "state_dir": str(autosell_state_dir())}
