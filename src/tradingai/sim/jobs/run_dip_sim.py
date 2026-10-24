"""Run the deterministic open-dip simulator over Alpaca 1m history."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from tradingai.sim.config import (
    DEFAULT_SIM_CONFIG_PATH,
    clear_sim_config_cache,
    get_charts_dir,
    get_export_charts,
    get_query_latency_seconds,
    get_query_pace_seconds,
    get_scan_batch_size,
    get_sim_buy_notional,
    get_sim_database_url,
    get_sim_dip_pct,
    get_sim_entry_cutoff,
    get_sim_exclude_first_bar_green,
    get_sim_max_positions,
    get_sim_min_bounce_score,
    get_sim_profit_taker_buffer,
    get_sim_require_status,
    get_sim_sell_trail,
    get_sim_start_cash,
    get_sim_stop_loss,
    get_sim_target_return,
    get_sim_use_trail,
    get_sim_window_minutes,
)
from tradingai.sim.engine import DipSimEngine
from tradingai.sim.strategy import StrategyParams


def _parse_status(raw: str) -> tuple[str, ...]:
    parts = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    allowed = {"reclaiming", "stabilizing", "plunging"}
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Invalid status values: {', '.join(bad)}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
    if not parts:
        return ("reclaiming",)
    return tuple(parts)


def run_dip_sim(
    *,
    tickers: list[str] | None = None,
    dip_pct: float | None = None,
    window_minutes: int | None = None,
    require_status: tuple[str, ...] | None = None,
    target_return: float | None = None,
    stop_loss: float | None = None,
    sell_trail: float | None = None,
    profit_taker_buffer: float | None = None,
    buy_notional: float | None = None,
    max_positions: int | None = None,
    start_cash: float | None = None,
    exclude_first_bar_green: bool | None = None,
    min_bounce_score: float | None = None,
    use_trail: bool | None = None,
    entry_cutoff_hour: int | None = None,
    entry_cutoff_minute: int | None = None,
    start_day: date | None = None,
    end_day: date | None = None,
    database_url: str | None = None,
    persist: bool = True,
    config_path: str | None = None,
    query_latency_seconds: float | None = None,
    query_pace_seconds: float | None = None,
    scan_batch_size: int | None = None,
    export_charts: bool | None = None,
    charts_dir: str | None = None,
) -> dict:
    """Defaults come from ``dip_sim.toml`` (or ``config_path``), not from ``.env``."""
    cfg = config_path
    cutoff_h, cutoff_m = get_sim_entry_cutoff(config_path=cfg)
    params = StrategyParams(
        dip_pct=float(dip_pct) if dip_pct is not None else get_sim_dip_pct(config_path=cfg),
        window_minutes=(
            int(window_minutes)
            if window_minutes is not None
            else get_sim_window_minutes(config_path=cfg)
        ),
        require_status=(
            require_status
            if require_status is not None
            else get_sim_require_status(config_path=cfg)
        ),
        target_return=(
            float(target_return)
            if target_return is not None
            else get_sim_target_return(config_path=cfg)
        ),
        stop_loss=(
            float(stop_loss) if stop_loss is not None else get_sim_stop_loss(config_path=cfg)
        ),
        sell_trail=(
            sell_trail
            if sell_trail is not None
            else get_sim_sell_trail(config_path=cfg)
        ),
        profit_taker_buffer=(
            float(profit_taker_buffer)
            if profit_taker_buffer is not None
            else get_sim_profit_taker_buffer(config_path=cfg)
        ),
        buy_notional_usd=(
            float(buy_notional)
            if buy_notional is not None
            else get_sim_buy_notional(config_path=cfg)
        ),
        max_positions=(
            int(max_positions)
            if max_positions is not None
            else get_sim_max_positions(config_path=cfg)
        ),
        exclude_first_bar_green=(
            bool(exclude_first_bar_green)
            if exclude_first_bar_green is not None
            else get_sim_exclude_first_bar_green(config_path=cfg)
        ),
        min_bounce_score=(
            min_bounce_score
            if min_bounce_score is not None
            else get_sim_min_bounce_score(config_path=cfg)
        ),
        use_trail=(
            bool(use_trail) if use_trail is not None else get_sim_use_trail(config_path=cfg)
        ),
        entry_cutoff_hour=(
            int(entry_cutoff_hour) if entry_cutoff_hour is not None else cutoff_h
        ),
        entry_cutoff_minute=(
            int(entry_cutoff_minute) if entry_cutoff_minute is not None else cutoff_m
        ),
    )
    engine = DipSimEngine(
        params=params,
        start_cash=(
            float(start_cash) if start_cash is not None else get_sim_start_cash(config_path=cfg)
        ),
        database_url=database_url or get_sim_database_url(config_path=cfg),
        tickers=tickers,
        persist=persist,
        query_latency_seconds=(
            float(query_latency_seconds)
            if query_latency_seconds is not None
            else get_query_latency_seconds(config_path=cfg)
        ),
        query_pace_seconds=(
            float(query_pace_seconds)
            if query_pace_seconds is not None
            else get_query_pace_seconds(config_path=cfg)
        ),
        scan_batch_size=(
            int(scan_batch_size)
            if scan_batch_size is not None
            else get_scan_batch_size(config_path=cfg)
        ),
        export_charts=(
            bool(export_charts)
            if export_charts is not None
            else get_export_charts(config_path=cfg)
        ),
        charts_dir=charts_dir or get_charts_dir(config_path=cfg),
    )
    return engine.run(start_day=start_day, end_day=end_day)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run open-dip simulation over Alpaca 1m bars. "
            f"Defaults from {DEFAULT_SIM_CONFIG_PATH.name} (not .env)."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to sim TOML (default: {DEFAULT_SIM_CONFIG_PATH}).",
    )
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument(
        "--dip-pct",
        type=float,
        default=None,
        help="Override dip depth gate (%%) from config.",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=None,
        help="Override open-dip window after 09:30 ET.",
    )
    parser.add_argument(
        "--require-status",
        type=_parse_status,
        default=None,
        help="Comma-separated bounce statuses (default from config).",
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="Override net target return fraction.",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Override hard stop-loss fraction.",
    )
    parser.add_argument(
        "--sell-trail",
        type=float,
        default=None,
        help="Override trail distance fraction.",
    )
    parser.add_argument(
        "--profit-taker-buffer",
        type=float,
        default=None,
        help="Override profit-taker cushion (fraction).",
    )
    parser.add_argument("--buy-notional", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--start-cash", type=float, default=None)
    parser.set_defaults(exclude_first_bar_green=None)
    green_group = parser.add_mutually_exclusive_group()
    green_group.add_argument(
        "--exclude-first-bar-green",
        dest="exclude_first_bar_green",
        action="store_const",
        const=True,
        help="Skip names whose opening composite bar is green.",
    )
    green_group.add_argument(
        "--no-exclude-first-bar-green",
        dest="exclude_first_bar_green",
        action="store_const",
        const=False,
        help="Force-allow green opening bars (overrides config).",
    )
    parser.add_argument("--min-bounce-score", type=float, default=None)
    parser.add_argument(
        "--no-trail",
        action="store_true",
        help="Disable trailing stop (limit + hard stop only).",
    )
    parser.add_argument(
        "--entry-cutoff",
        default=None,
        help="HH:MM ET cutoff for new entries (default from config).",
    )
    parser.add_argument("--start-day", type=date.fromisoformat, default=None)
    parser.add_argument("--end-day", type=date.fromisoformat, default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write run/orders/fills to the sim DB.",
    )
    parser.add_argument(
        "--charts-dir",
        default=None,
        help="Directory for trade PNG charts (default from config).",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Disable trade chart export.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full run summary JSON (default: short summary only).",
    )
    args = parser.parse_args()

    clear_sim_config_cache()
    config_path = args.config

    cutoff_hour: int | None = None
    cutoff_minute: int | None = None
    if args.entry_cutoff is not None:
        try:
            hour_s, minute_s = args.entry_cutoff.split(":", 1)
            cutoff_hour = int(hour_s)
            cutoff_minute = int(minute_s)
        except Exception as exc:
            print(f"Invalid --entry-cutoff '{args.entry_cutoff}': {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    use_trail: bool | None = False if args.no_trail else None

    try:
        result = run_dip_sim(
            tickers=args.tickers,
            dip_pct=args.dip_pct,
            window_minutes=args.window_minutes,
            require_status=args.require_status,
            target_return=args.target_return,
            stop_loss=args.stop_loss,
            sell_trail=args.sell_trail,
            profit_taker_buffer=args.profit_taker_buffer,
            buy_notional=args.buy_notional,
            max_positions=args.max_positions,
            start_cash=args.start_cash,
            exclude_first_bar_green=args.exclude_first_bar_green,
            min_bounce_score=args.min_bounce_score,
            use_trail=use_trail,
            entry_cutoff_hour=cutoff_hour,
            entry_cutoff_minute=cutoff_minute,
            start_day=args.start_day,
            end_day=args.end_day,
            database_url=args.database_url,
            persist=not args.no_persist,
            config_path=config_path,
            export_charts=False if args.no_charts else None,
            charts_dir=args.charts_dir,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        charts = result.get("charts") or {}
        print(
            f"run_id={result.get('run_id')}  "
            f"trades={result.get('trade_count')}  "
            f"wins={result.get('win_count')}  "
            f"win_rate={float(result.get('win_rate') or 0):.2%}  "
            f"realized_pnl={float(result.get('realized_pnl') or 0):+.2f}  "
            f"end_cash={float(result.get('end_cash') or 0):.2f}  "
            f"charts={charts.get('charts_written', 0)}",
            flush=True,
        )
        if charts.get("out_dir"):
            print(f"charts_dir={charts['out_dir']}", flush=True)


if __name__ == "__main__":
    main()
