"""CLI: background autosell watcher (same sell args as tradingai-sell-trail)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from tradingai.ibkr.autosell import (
    DEFAULT_POLL_INTERVAL_SEC,
    AutosellConfig,
    job_status,
    make_job_id,
    run_watcher_loop,
    start_background_job,
    stop_job,
)
from tradingai.providers.ibkr_tws import DEFAULT_SELL_TRAIL_PERCENT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch IBKR TWS fills and trail-sell unpaired buys for a symbol. "
            "Same sell sizing args as tradingai-sell-trail. Runs as a background job "
            f"by default; polls every {DEFAULT_POLL_INTERVAL_SEC:g}s."
        )
    )
    parser.add_argument("symbol", nargs="?", help="Ticker, e.g. COIN.")
    parser.add_argument(
        "--qty",
        "--quantity",
        dest="quantity",
        type=float,
        default=None,
        help="Optional max shares per autosell (default: exact unpaired buy qty).",
    )
    parser.add_argument(
        "--price",
        "--limit-price",
        dest="limit_price",
        type=float,
        default=None,
        help="Trail stop / limit reference (default: fee-aware target on buy).",
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help=(
            "Net profit fraction after fees (e.g. 0.01 = 1%%). "
            "Used when --price is omitted."
        ),
    )
    parser.add_argument(
        "--trail-percent",
        type=float,
        default=DEFAULT_SELL_TRAIL_PERCENT,
        help=(
            "IB trailing percent points "
            f"(default {DEFAULT_SELL_TRAIL_PERCENT})."
        ),
    )
    parser.add_argument("--tif", default="DAY", help="Time in force (default DAY).")
    parser.add_argument(
        "--outside-rth",
        action="store_true",
        help="Allow outside regular trading hours.",
    )
    parser.add_argument("--account", default=None, help="Optional IB account code.")
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually send TRAIL LIMIT sells (requires TRADINGAI_IBKR_ALLOW_ORDERS=1). "
            "Without this flag autosell dry-runs sells only."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --submit when starting (skips interactive confirm).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"Poll interval seconds (default {DEFAULT_POLL_INTERVAL_SEC:g}).",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run the poll loop in this process (used by the background worker).",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Job id (default: SYMBOL_UTC-timestamp). Required with --foreground.",
    )
    parser.add_argument(
        "--status",
        nargs="?",
        const="",
        default=None,
        metavar="JOB_ID",
        help="Print job status (optional JOB_ID; omit to list all).",
    )
    parser.add_argument(
        "--stop",
        metavar="JOB_ID",
        default=None,
        help="Stop a running autosell job.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.status is not None:
        job_id = args.status or None
        print(json.dumps(job_status(job_id), indent=2, default=str))
        return

    if args.stop:
        print(json.dumps(stop_job(args.stop), indent=2, default=str))
        return

    if not args.symbol:
        parser.error("symbol is required unless using --status / --stop")

    if args.submit and not args.yes and not args.foreground:
        reply = input(
            f"Start LIVE autosell watcher for {args.symbol.upper()} "
            f"(interval={args.interval:g}s, trail={args.trail_percent}%)? [y/N] "
        ).strip().lower()
        if reply not in {"y", "yes"}:
            print(json.dumps({"cancelled": True}, indent=2))
            raise SystemExit(0)

    config = AutosellConfig(
        symbol=args.symbol,
        quantity=args.quantity,
        limit_price=args.limit_price,
        target_return=args.target_return,
        trail_percent=args.trail_percent,
        tif=args.tif,
        outside_rth=args.outside_rth,
        account=args.account,
        submit=bool(args.submit),
        interval_sec=float(args.interval),
    )

    if args.foreground:
        job_id = args.job_id or make_job_id(config.normalized_symbol())
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "mode": "foreground",
                    "config": asdict(config),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        run_watcher_loop(job_id, config)
        return

    result = start_background_job(config, job_id=args.job_id)
    print(json.dumps(result, indent=2, default=str))
    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
