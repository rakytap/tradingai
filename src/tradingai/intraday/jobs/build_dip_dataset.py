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
from tradingai.intraday.dip_dataset import (
    DEFAULT_CUTS_AFTER,
    DEFAULT_CUTS_BEFORE,
    DEFAULT_CUT_VICINITY_MINUTES,
    DEFAULT_DATASET_DIP_PCT,
    DEFAULT_DATASET_RISE_PCT,
    DEFAULT_LABEL_HORIZON_MINUTES,
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


def build_dip_dataset(
    tickers: list[str] | None = None,
    *,
    source: str = SOURCE_HISTORY,
    database_url: str | None = None,
    dip_pct: float = DEFAULT_DATASET_DIP_PCT,
    rise_pct: float = DEFAULT_DATASET_RISE_PCT,
    cuts_before: int = DEFAULT_CUTS_BEFORE,
    cuts_after: int = DEFAULT_CUTS_AFTER,
    min_horizon_minutes: int = DEFAULT_MIN_HORIZON_MINUTES,
    label_horizon_minutes: int = DEFAULT_LABEL_HORIZON_MINUTES,
    cut_vicinity_minutes: int = DEFAULT_CUT_VICINITY_MINUTES,
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
            ticker_sessions = split_rth_sessions(history)
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
    samples, bars, qualifying = assemble_cuts(
        sessions,
        dip_pct=dip_pct,
        rise_pct=rise_pct,
        cuts_before=cuts_before,
        cuts_after=cuts_after,
        min_horizon_minutes=min_horizon_minutes,
        label_horizon_minutes=label_horizon_minutes,
        cut_vicinity_minutes=cut_vicinity_minutes,
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
        "sessions_seen": sessions_seen,
        "sessions_with_prior_close": len(sessions),
        "sessions_qualifying": qualifying,
        "samples": int(len(samples)),
        "skipped_no_prior_close": skipped_no_prior_close,
        "dip_pct": float(dip_pct),
        "rise_pct": float(rise_pct),
        "cuts_before": int(cuts_before),
        "cuts_after": int(cuts_after),
        "min_horizon_minutes": int(min_horizon_minutes),
        "label_horizon_minutes": int(label_horizon_minutes),
        "cut_vicinity_minutes": int(cut_vicinity_minutes),
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
    parser.add_argument("--dip-pct", type=float, default=DEFAULT_DATASET_DIP_PCT)
    parser.add_argument(
        "--rise-pct",
        type=float,
        default=DEFAULT_DATASET_RISE_PCT,
        help="Legacy sample field; labeling uses the fee-aware sell target.",
    )
    parser.add_argument("--cuts-before", type=int, default=DEFAULT_CUTS_BEFORE)
    parser.add_argument("--cuts-after", type=int, default=DEFAULT_CUTS_AFTER)
    parser.add_argument(
        "--min-horizon-minutes",
        type=int,
        default=DEFAULT_MIN_HORIZON_MINUTES,
        help=(
            "Require at least this many post-cut bars when sampling cuts "
            f"(default {DEFAULT_MIN_HORIZON_MINUTES}; raised to label horizon)."
        ),
    )
    parser.add_argument(
        "--label-horizon-minutes",
        type=int,
        default=DEFAULT_LABEL_HORIZON_MINUTES,
        help=(
            "Minutes after the cut in which hitting the sell target counts as "
            f"label=1 (default {DEFAULT_LABEL_HORIZON_MINUTES})."
        ),
    )
    parser.add_argument(
        "--cut-vicinity-minutes",
        type=int,
        default=DEFAULT_CUT_VICINITY_MINUTES,
        help=(
            "Only sample cut ends within this many minutes of the dip "
            f"(default {DEFAULT_CUT_VICINITY_MINUTES}; 0 = whole session)."
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
    try:
        result = build_dip_dataset(
            args.tickers,
            source=args.source,
            database_url=args.database_url,
            dip_pct=args.dip_pct,
            rise_pct=args.rise_pct,
            cuts_before=args.cuts_before,
            cuts_after=args.cuts_after,
            min_horizon_minutes=args.min_horizon_minutes,
            label_horizon_minutes=args.label_horizon_minutes,
            cut_vicinity_minutes=args.cut_vicinity_minutes,
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
