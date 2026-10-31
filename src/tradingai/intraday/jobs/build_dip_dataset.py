"""Build labeled dip-recovery training cuts from IBKR history or Alpaca sim bars."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from tradingai.data.connection import initialize_schema
from tradingai.data.jobs.progress import log_done, log_error, log_progress, log_start
from tradingai.data.repositories.daily_prices import list_closes
from tradingai.data.repositories.intraday_bars import (
    INTRADAY_HISTORY_TABLE,
    list_intraday_tickers,
    load_intraday_bars,
)
from tradingai.intraday.bars import prior_daily_close, rth_open_for_day
from tradingai.intraday.dip_charts import pick_stratified_samples, write_dip_dataset_viz
from tradingai.intraday.config import (
    clear_intraday_config_cache,
    get_buy_notional_usd,
    get_label_horizon_minutes,
    get_label_stop_loss,
    get_target_return,
)
from tradingai.intraday.dip_dataset import (
    DEFAULT_CUTS_PER_SESSION,
    DEFAULT_CUT_MAX_MINUTES_FROM_OPEN,
    DEFAULT_CUT_MIN_MINUTES_FROM_OPEN,
    DEFAULT_DATASET_DIP_PCT,
    DEFAULT_DATASET_RISE_PCT,
    DEFAULT_MIN_HORIZON_MINUTES,
    DEFAULT_VALLEY_ORDER_MINUTES,
    DEFAULT_VALLEY_WINDOW_MINUTES,
    assemble_cuts,
    split_rth_sessions,
    write_dip_dataset,
)
from tradingai.sim.config import get_sim_database_url
from tradingai.sim.repository import (
    initialize_sim_schema,
    list_sim_tickers,
    load_prior_close as load_sim_prior_close,
    load_sim_bars_1m,
)

DEFAULT_OUT_DIR = Path("data/dip_dataset")
DEFAULT_ALPACA_OUT_DIR = Path("data/dip_dataset_alpaca")
DEFAULT_VIZ_COUNT = 20
SOURCE_HISTORY = "history"
SOURCE_SIM = "sim"
SOURCE_ALPACA = "alpaca"  # alias for sim (Alpaca-backed tradingai_sim)


def resolve_prior_close(
    ticker: str,
    session_day: date,
    daily_closes: list[tuple[date, float]],
    history: pd.DataFrame,
) -> float | None:
    """Prior close from daily prices, else the previous stored session, else the provider."""
    earlier = [close for day, close in daily_closes if day < session_day]
    if earlier:
        return float(earlier[-1])
    if history is not None and not history.empty:
        open_ts = rth_open_for_day(session_day)
        previous = history.loc[history.index < open_ts, "close"].dropna()
        if not previous.empty:
            return float(previous.iloc[-1])
    return prior_daily_close(ticker, before=session_day)


def resolve_sim_prior_close(
    ticker: str,
    session_day: date,
    history: pd.DataFrame,
    *,
    database_url: str | None = None,
) -> float | None:
    """Prior close from ``sim_daily_closes``, else the previous sim session close."""
    prior = load_sim_prior_close(ticker, session_day, database_url=database_url)
    if prior is not None and prior > 0:
        return float(prior)
    if history is not None and not history.empty:
        open_ts = rth_open_for_day(session_day)
        previous = history.loc[history.index < open_ts, "close"].dropna()
        if not previous.empty:
            return float(previous.iloc[-1])
    return None


def _normalize_source(source: str) -> str:
    raw = str(source or SOURCE_HISTORY).strip().lower()
    if raw in {SOURCE_SIM, SOURCE_ALPACA}:
        return SOURCE_SIM
    if raw == SOURCE_HISTORY:
        return SOURCE_HISTORY
    raise ValueError(f"Unsupported source {source!r}; use history, sim, or alpaca.")


def take_recent_sessions(
    sessions: list[tuple[date, pd.DataFrame]],
    days: int | None,
) -> list[tuple[date, pd.DataFrame]]:
    """Keep the most recent ``days`` RTH sessions (chronological order preserved).

    ``days is None`` or ``days <= 0`` means keep all sessions.
    """
    if days is None or int(days) <= 0:
        return list(sessions)
    n = int(days)
    if len(sessions) <= n:
        return list(sessions)
    return list(sessions[-n:])


def build_dip_dataset(
    tickers: list[str] | None = None,
    *,
    source: str = SOURCE_HISTORY,
    database_url: str | None = None,
    days: int | None = None,
    dip_pct: float = DEFAULT_DATASET_DIP_PCT,
    rise_pct: float = DEFAULT_DATASET_RISE_PCT,
    n_cuts: int = DEFAULT_CUTS_PER_SESSION,
    min_horizon_minutes: int | None = None,
    label_horizon_minutes: int | None = None,
    stop_loss: float | None = None,
    target_return: float | None = None,
    buy_notional_usd: float | None = None,
    cut_min_minutes_from_open: int = DEFAULT_CUT_MIN_MINUTES_FROM_OPEN,
    cut_max_minutes_from_open: int = DEFAULT_CUT_MAX_MINUTES_FROM_OPEN,
    valley_window_minutes: int = DEFAULT_VALLEY_WINDOW_MINUTES,
    valley_order_minutes: int = DEFAULT_VALLEY_ORDER_MINUTES,
    seed: int = 0,
    out_dir: Path | str | None = None,
    viz_count: int = DEFAULT_VIZ_COUNT,
) -> dict:
    source_key = _normalize_source(source)
    if out_dir is None:
        out_dir = DEFAULT_ALPACA_OUT_DIR if source_key == SOURCE_SIM else DEFAULT_OUT_DIR
    if source_key == SOURCE_SIM:
        initialize_sim_schema(database_url)
        db_url = database_url or get_sim_database_url()
        if tickers:
            universe = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker.strip()))
        else:
            universe = list_sim_tickers(database_url=db_url)
    else:
        initialize_schema()
        db_url = None
        if tickers:
            universe = list(dict.fromkeys(ticker.upper() for ticker in tickers if ticker.strip()))
        else:
            universe = list_intraday_tickers(table=INTRADAY_HISTORY_TABLE)

    log_start("tradingai-build-dip-dataset", len(universe))
    sessions: list[tuple[str, date, pd.DataFrame, float]] = []
    sessions_seen = 0
    skipped_no_prior_close = 0
    errors: list[str] = []
    for index, ticker in enumerate(universe, start=1):
        try:
            if source_key == SOURCE_SIM:
                history = load_sim_bars_1m(ticker, database_url=db_url)
                daily: list[tuple[date, float]] = []
            else:
                history = load_intraday_bars(ticker, table=INTRADAY_HISTORY_TABLE)
                daily = list_closes(ticker)
            ticker_sessions = take_recent_sessions(
                split_rth_sessions(history), days
            )
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            log_error(index, len(universe), ticker, str(exc))
            continue
        kept = 0
        for day, bars in ticker_sessions:
            sessions_seen += 1
            if source_key == SOURCE_SIM:
                prior = resolve_sim_prior_close(
                    ticker, day, history, database_url=db_url
                )
            else:
                prior = resolve_prior_close(ticker, day, daily, history)
            if prior is None or prior <= 0:
                skipped_no_prior_close += 1
                continue
            sessions.append((ticker, day, bars, prior))
            kept += 1
        log_progress(index, len(universe), ticker, f"{kept} sessions")
    print(
        f"assembling cuts from {len(sessions)} sessions "
        f"(this can take a few minutes)...",
        flush=True,
    )
    resolved_horizon = int(
        label_horizon_minutes
        if label_horizon_minutes is not None
        else get_label_horizon_minutes()
    )
    resolved_min_horizon = int(
        min_horizon_minutes
        if min_horizon_minutes is not None
        else max(DEFAULT_MIN_HORIZON_MINUTES, resolved_horizon)
    )
    resolved_stop = float(
        stop_loss if stop_loss is not None else get_label_stop_loss()
    )
    resolved_return = float(
        target_return if target_return is not None else get_target_return()
    )
    resolved_notional = float(
        buy_notional_usd
        if buy_notional_usd is not None
        else get_buy_notional_usd()
    )
    samples, bars, qualifying = assemble_cuts(
        sessions,
        dip_pct=dip_pct,
        rise_pct=rise_pct,
        n_cuts=n_cuts,
        min_horizon_minutes=resolved_min_horizon,
        label_horizon_minutes=resolved_horizon,
        stop_loss=resolved_stop,
        target_return=resolved_return,
        buy_notional_usd=resolved_notional,
        cut_min_minutes_from_open=cut_min_minutes_from_open,
        cut_max_minutes_from_open=cut_max_minutes_from_open,
        valley_window_minutes=valley_window_minutes,
        valley_order_minutes=valley_order_minutes,
        seed=seed,
    )
    print(
        f"assembled {len(samples)} samples / {len(bars)} cut bars "
        f"from {qualifying} qualifying sessions; writing parquet...",
        flush=True,
    )
    destination = write_dip_dataset(samples, bars, out_dir)
    print(f"wrote {destination}", flush=True)
    viz_dir = None
    if int(viz_count) > 0 and not samples.empty:
        # Only keep session frames needed for the stratified viz sample.
        selected = pick_stratified_samples(samples, n=int(viz_count), seed=seed)
        needed = {
            (str(row.ticker).upper(), pd.Timestamp(row.session_date).date())
            for row in selected.itertuples(index=False)
        }
        session_map = {
            (str(ticker).upper(), day): frame
            for ticker, day, frame, _prior in sessions
            if (str(ticker).upper(), day) in needed
        }
        print(f"writing {len(selected)} sample charts...", flush=True)
        viz_dir = write_dip_dataset_viz(
            samples,
            bars,
            Path(destination) / "viz",
            sessions=session_map,
            n=int(viz_count),
            seed=seed,
        )
        print(f"wrote viz → {viz_dir}", flush=True)
    log_done(
        "tradingai-build-dip-dataset",
        qualifying,
        len(universe),
        samples=len(samples),
    )
    return {
        "source": source_key,
        "database_url": db_url,
        "tickers": universe,
        "days": None if days is None or int(days) <= 0 else int(days),
        "sessions_seen": sessions_seen,
        "sessions_with_prior_close": len(sessions),
        "sessions_qualifying": qualifying,
        "samples": int(len(samples)),
        "skipped_no_prior_close": skipped_no_prior_close,
        "dip_pct": float(dip_pct),
        "rise_pct": float(rise_pct),
        "n_cuts": int(n_cuts),
        "min_horizon_minutes": (
            None if min_horizon_minutes is None else int(min_horizon_minutes)
        ),
        "label_horizon_minutes": resolved_horizon,
        "stop_loss": resolved_stop,
        "target_return": resolved_return,
        "buy_notional_usd": resolved_notional,
        "cut_min_minutes_from_open": int(cut_min_minutes_from_open),
        "cut_max_minutes_from_open": int(cut_max_minutes_from_open),
        "valley_window_minutes": int(valley_window_minutes),
        "valley_order_minutes": int(valley_order_minutes),
        "seed": int(seed),
        "out_dir": str(destination),
        "viz_count": int(viz_count),
        "viz_dir": None if viz_dir is None else str(viz_dir),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write labeled dip-recovery training cuts from intraday_history_bars "
            "or Alpaca-backed sim_bars_1m (tradingai_sim)."
        )
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument(
        "--source",
        choices=(SOURCE_HISTORY, SOURCE_SIM, SOURCE_ALPACA),
        default=SOURCE_HISTORY,
        help=(
            "Bar source: history (IBKR intraday_history_bars) or "
            "sim/alpaca (tradingai_sim.sim_bars_1m)."
        ),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override sim DB URL when --source is sim/alpaca.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "Use only the most recent N RTH sessions per ticker "
            "(default: all available sessions in the bar store)."
        ),
    )
    parser.add_argument("--dip-pct", type=float, default=DEFAULT_DATASET_DIP_PCT)
    parser.add_argument(
        "--rise-pct",
        type=float,
        default=DEFAULT_DATASET_RISE_PCT,
        help="Legacy sample field; labeling uses the fee-aware sell target.",
    )
    parser.add_argument(
        "--cuts",
        type=int,
        default=DEFAULT_CUTS_PER_SESSION,
        help=(
            "Random cuts per qualifying session inside the open-relative "
            f"window (default {DEFAULT_CUTS_PER_SESSION})."
        ),
    )
    parser.add_argument(
        "--min-horizon-minutes",
        type=int,
        default=None,
        help=(
            "Require at least this many post-cut bars when sampling cuts "
            "(default: max of hardcoded floor and label horizon / env)."
        ),
    )
    parser.add_argument(
        "--label-horizon-minutes",
        type=int,
        default=None,
        help=(
            "Minutes after the cut for stop/target labeling "
            "(default TRADINGAI_DIP_LABEL_HORIZON_MINUTES / 15)."
        ),
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help=(
            "Fraction below cut close that forces label=0 "
            "(default TRADINGAI_DIP_LABEL_STOP_LOSS / 0.006). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help=(
            "Net profit fraction for fee-aware sell target "
            "(default TRADINGAI_OPEN_DIP_TARGET_RETURN)."
        ),
    )
    parser.add_argument(
        "--buy-notional",
        type=float,
        default=None,
        help=(
            "Notional USD for fee estimate when pricing the sell target "
            "(default TRADINGAI_OPEN_DIP_BUY_NOTIONAL / 20000)."
        ),
    )
    parser.add_argument(
        "--cut-min-minutes-from-open",
        type=int,
        default=DEFAULT_CUT_MIN_MINUTES_FROM_OPEN,
        help=(
            "Earliest cut: minutes after session open "
            f"(default {DEFAULT_CUT_MIN_MINUTES_FROM_OPEN} = 1.0h)."
        ),
    )
    parser.add_argument(
        "--cut-max-minutes-from-open",
        type=int,
        default=DEFAULT_CUT_MAX_MINUTES_FROM_OPEN,
        help=(
            "Latest cut: minutes after session open "
            f"(default {DEFAULT_CUT_MAX_MINUTES_FROM_OPEN} = 2.5h)."
        ),
    )
    parser.add_argument(
        "--valley-window-minutes",
        type=int,
        default=DEFAULT_VALLEY_WINDOW_MINUTES,
        help=(
            "Search for the most pronounced valley in the first N RTH minutes "
            f"(default {DEFAULT_VALLEY_WINDOW_MINUTES})."
        ),
    )
    parser.add_argument(
        "--valley-order-minutes",
        type=int,
        default=DEFAULT_VALLEY_ORDER_MINUTES,
        help=(
            "Local-min neighborhood half-width in minutes "
            f"(default {DEFAULT_VALLEY_ORDER_MINUTES})."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            f"Output directory (default {DEFAULT_OUT_DIR} for history, "
            f"{DEFAULT_ALPACA_OUT_DIR} for sim/alpaca)."
        ),
    )
    parser.add_argument(
        "--viz-count",
        type=int,
        default=DEFAULT_VIZ_COUNT,
        help="Stratified sample charts to write under out/viz (0 disables).",
    )
    args = parser.parse_args()
    clear_intraday_config_cache()
    try:
        result = build_dip_dataset(
            args.tickers,
            source=args.source,
            database_url=args.database_url,
            days=args.days,
            dip_pct=args.dip_pct,
            rise_pct=args.rise_pct,
            n_cuts=args.cuts,
            min_horizon_minutes=args.min_horizon_minutes,
            label_horizon_minutes=args.label_horizon_minutes,
            stop_loss=args.stop_loss,
            target_return=args.target_return,
            buy_notional_usd=args.buy_notional,
            cut_min_minutes_from_open=args.cut_min_minutes_from_open,
            cut_max_minutes_from_open=args.cut_max_minutes_from_open,
            valley_window_minutes=args.valley_window_minutes,
            valley_order_minutes=args.valley_order_minutes,
            seed=args.seed,
            out_dir=args.out,
            viz_count=args.viz_count,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
