"""Build labeled dip-recovery training cuts from ``intraday_history_bars``."""

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
from tradingai.intraday.dip_dataset import (
    DEFAULT_CUTS_AFTER,
    DEFAULT_CUTS_BEFORE,
    DEFAULT_DATASET_DIP_PCT,
    DEFAULT_DATASET_RISE_PCT,
    DEFAULT_MIN_HORIZON_MINUTES,
    assemble_cuts,
    split_rth_sessions,
    write_dip_dataset,
)

DEFAULT_OUT_DIR = Path("data/dip_dataset")


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


def build_dip_dataset(
    tickers: list[str] | None = None,
    *,
    dip_pct: float = DEFAULT_DATASET_DIP_PCT,
    rise_pct: float = DEFAULT_DATASET_RISE_PCT,
    cuts_before: int = DEFAULT_CUTS_BEFORE,
    cuts_after: int = DEFAULT_CUTS_AFTER,
    min_horizon_minutes: int = DEFAULT_MIN_HORIZON_MINUTES,
    seed: int = 0,
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> dict:
    initialize_schema()
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
            prior = resolve_prior_close(ticker, day, daily, history)
            if prior is None or prior <= 0:
                skipped_no_prior_close += 1
                continue
            sessions.append((ticker, day, bars, prior))
            kept += 1
        log_progress(index, len(universe), ticker, f"{kept} sessions")
    samples, bars, qualifying = assemble_cuts(
        sessions,
        dip_pct=dip_pct,
        rise_pct=rise_pct,
        cuts_before=cuts_before,
        cuts_after=cuts_after,
        min_horizon_minutes=min_horizon_minutes,
        seed=seed,
    )
    destination = write_dip_dataset(samples, bars, out_dir)
    log_done(
        "tradingai-build-dip-dataset",
        qualifying,
        len(universe),
        samples=len(samples),
    )
    return {
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
        "seed": int(seed),
        "out_dir": str(destination),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write labeled dip-recovery training cuts from intraday_history_bars."
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument("--dip-pct", type=float, default=DEFAULT_DATASET_DIP_PCT)
    parser.add_argument(
        "--rise-pct",
        type=float,
        default=DEFAULT_DATASET_RISE_PCT,
        help="Label threshold in percent above the cut-end close (default 0.4).",
    )
    parser.add_argument("--cuts-before", type=int, default=DEFAULT_CUTS_BEFORE)
    parser.add_argument("--cuts-after", type=int, default=DEFAULT_CUTS_AFTER)
    parser.add_argument(
        "--min-horizon-minutes",
        type=int,
        default=DEFAULT_MIN_HORIZON_MINUTES,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    try:
        result = build_dip_dataset(
            args.tickers,
            dip_pct=args.dip_pct,
            rise_pct=args.rise_pct,
            cuts_before=args.cuts_before,
            cuts_after=args.cuts_after,
            min_horizon_minutes=args.min_horizon_minutes,
            seed=args.seed,
            out_dir=args.out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
