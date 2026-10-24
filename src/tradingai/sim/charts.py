"""Export session charts for sim buys (filled wins/losses and canceled pendings)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tradingai.intraday.bars import filter_rth_session
from tradingai.intraday.config import ET
from tradingai.sim.broker import ClosedTrade, WatchlistEntry
from tradingai.sim.repository import load_sim_bars_1m

DEFAULT_CHARTS_DIR = Path("data/sim_charts")


@dataclass(frozen=True)
class ChartCase:
    """One buy attempt to plot (filled round-trip or canceled pending)."""

    ticker: str
    session_day: date
    outcome: str  # win | loss | canceled
    submit_at: datetime | None = None
    submit_price: float | None = None
    fill_at: datetime | None = None
    fill_price: float | None = None
    exit_at: datetime | None = None
    exit_price: float | None = None
    cancel_at: datetime | None = None
    exit_reason: str | None = None
    realized_pnl: float | None = None

    @property
    def label(self) -> str:
        bits = [self.ticker, self.session_day.isoformat(), self.outcome]
        if self.exit_reason:
            bits.append(self.exit_reason)
        return "_".join(bits)


def _as_et(ts: datetime | pd.Timestamp) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(ET)
    else:
        t = t.tz_convert(ET)
    return t.to_pydatetime()


def _session_day_of(ts: datetime) -> date:
    return _as_et(ts).date()


def collect_chart_cases(
    *,
    trades: list[ClosedTrade],
    watchlist: list[WatchlistEntry],
) -> list[ChartCase]:
    """Build chart cases for closed trades (win/loss) and canceled pendings."""
    cases: list[ChartCase] = []

    # Match filled watch rows to trades by ticker + fill time proximity.
    filled_watch = [w for w in watchlist if w.status == "filled"]
    used_watch_ids: set[int] = set()

    for trade in trades:
        session_day = _session_day_of(trade.entry_time)
        outcome = "win" if trade.realized_pnl > 0 else "loss"
        submit_at = None
        submit_price = None
        for w in filled_watch:
            if w.watch_id in used_watch_ids:
                continue
            if w.ticker != trade.ticker:
                continue
            if w.filled_at is None:
                continue
            if abs((_as_et(w.filled_at) - _as_et(trade.entry_time)).total_seconds()) <= 1.0:
                submit_at = w.submitted_at
                submit_price = w.buy_limit_price
                used_watch_ids.add(w.watch_id)
                break
        cases.append(
            ChartCase(
                ticker=trade.ticker,
                session_day=session_day,
                outcome=outcome,
                submit_at=submit_at,
                submit_price=submit_price,
                fill_at=trade.entry_time,
                fill_price=trade.entry_price,
                exit_at=trade.exit_time,
                exit_price=trade.exit_price,
                exit_reason=trade.exit_reason,
                realized_pnl=trade.realized_pnl,
            )
        )

    for w in watchlist:
        if w.status != "canceled":
            continue
        cases.append(
            ChartCase(
                ticker=w.ticker,
                session_day=_session_day_of(w.submitted_at),
                outcome="canceled",
                submit_at=w.submitted_at,
                submit_price=w.buy_limit_price,
                cancel_at=w.canceled_at,
            )
        )

    cases.sort(key=lambda c: (c.session_day, c.ticker, c.outcome))
    return cases


def plot_trade_case(
    case: ChartCase,
    bars: pd.DataFrame,
    *,
    out_path: Path,
) -> Path:
    """Plot RTH close with transaction markers; write PNG to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = filter_rth_session(bars, session_day=case.session_day) if not bars.empty else bars
    fig, ax = plt.subplots(figsize=(11, 5))

    if session.empty:
        ax.set_title(f"{case.ticker} {case.session_day} — no bars")
        ax.text(0.5, 0.5, "No RTH bars", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(
            session.index,
            session["close"].astype(float),
            color="#1f77b4",
            linewidth=1.2,
            label="close",
        )
        # Light high/low band for context (not a candle chart).
        ax.fill_between(
            session.index,
            session["low"].astype(float),
            session["high"].astype(float),
            color="#1f77b4",
            alpha=0.12,
            label="high-low",
        )

        def _mark(ts: datetime | None, price: float | None, *, color: str, marker: str, label: str) -> None:
            if ts is None:
                return
            t = _as_et(ts)
            y = float(price) if price is not None else None
            if y is None:
                # Snap to nearest bar close.
                idx = session.index.get_indexer([pd.Timestamp(t)], method="nearest")
                if len(idx) and idx[0] >= 0:
                    y = float(session["close"].iloc[idx[0]])
                    t = session.index[idx[0]].to_pydatetime()
                else:
                    return
            ax.scatter([t], [y], color=color, marker=marker, s=80, zorder=5, label=label)
            ax.axvline(t, color=color, linestyle="--", alpha=0.45, linewidth=1.0)

        _mark(case.submit_at, case.submit_price, color="#ff7f0e", marker="v", label="submit")
        _mark(case.fill_at, case.fill_price, color="#2ca02c", marker="^", label="buy fill")
        _mark(case.exit_at, case.exit_price, color="#d62728", marker="x", label="sell/exit")
        _mark(case.cancel_at, case.submit_price, color="#7f7f7f", marker="o", label="cancel")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ET))
        ax.set_xlabel("Time (ET)")
        ax.set_ylabel("Price")

    pnl_bit = ""
    if case.realized_pnl is not None:
        pnl_bit = f"  pnl={case.realized_pnl:+.2f}"
    reason_bit = f"  ({case.exit_reason})" if case.exit_reason else ""
    ax.set_title(f"{case.ticker}  {case.session_day}  {case.outcome}{reason_bit}{pnl_bit}")
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate legend labels
    seen: set[str] = set()
    uniq = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            uniq.append((h, lab))
    if uniq:
        ax.legend([h for h, _ in uniq], [lab for _, lab in uniq], loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def export_trade_charts(
    *,
    trades: list[ClosedTrade],
    watchlist: list[WatchlistEntry],
    database_url: str | None,
    out_dir: Path | str,
    run_id: int | None = None,
) -> dict:
    """Write one PNG per buy attempt under ``out_dir`` / ``run_<id>``."""
    base = Path(out_dir)
    run_folder = base / (f"run_{run_id}" if run_id is not None else f"run_{datetime.now(ET).strftime('%Y%m%d_%H%M%S')}")
    run_folder.mkdir(parents=True, exist_ok=True)

    cases = collect_chart_cases(trades=trades, watchlist=watchlist)
    written: list[str] = []
    errors: list[str] = []

    for i, case in enumerate(cases, start=1):
        try:
            bars = load_sim_bars_1m(
                case.ticker,
                session_day=case.session_day,
                database_url=database_url,
            )
            fname = f"{i:04d}_{case.session_day.isoformat()}_{case.ticker}_{case.outcome}.png"
            path = plot_trade_case(case, bars, out_path=run_folder / fname)
            written.append(str(path))
        except Exception as exc:
            errors.append(f"{case.ticker} {case.session_day}: {exc}")

    manifest = {
        "out_dir": str(run_folder),
        "charts_written": len(written),
        "charts": written,
        "errors": errors[:25],
        "wins": sum(1 for c in cases if c.outcome == "win"),
        "losses": sum(1 for c in cases if c.outcome == "loss"),
        "canceled": sum(1 for c in cases if c.outcome == "canceled"),
    }
    (run_folder / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
