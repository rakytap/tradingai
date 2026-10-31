"""CLI: budget LIMIT buy + background trail (0.6% → 0.05% at target)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from tradingai.ibkr.buy_trail import (
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_TIGHT_POLL_INTERVAL_SEC,
    BuyTrailConfig,
    job_status,
    run_watcher_loop,
    start_buy_trail,
    stop_job,
)
from tradingai.intraday.config import get_buy_notional_usd
from tradingai.providers.ibkr_tws import (
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_BUY_TRAIL_1_5X_PERCENT,
    DEFAULT_BUY_TRAIL_2X_PERCENT,
    DEFAULT_BUY_TRAIL_PERCENT,
    DEFAULT_BUY_TRAIL_TIGHT_PERCENT,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Place a budget LIMIT buy (no STP; default mid from TWS), then "
            "background-monitor fills every 10s (5s after first tighten). "
            "After the buy is fully filled, place a TRAIL LIMIT at 0.6% from "
            "activate mid; tighten to 0.05%% / 0.03%% / 0.01%% when mid hits "
            "1× / 1.5× / 2× fee-aware TRADINGAI_OPEN_DIP_TARGET_RETURN. "
            "Exits when the position is flat."
        )
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Ticker, e.g. COIN (required unless --status/--stop).",
    )
    parser.add_argument(
        "--budget",
        "--total-budget",
        dest="total_budget",
        type=float,
        default=None,
        help=(
            "Gross cash budget in USD "
            f"(default TRADINGAI_OPEN_DIP_BUY_NOTIONAL / {get_buy_notional_usd():.0f})."
        ),
    )
    parser.add_argument(
        "--price",
        "--limit-price",
        dest="limit_price",
        type=float,
        default=None,
        help="Buy limit / trail activate price (default: current IB mid).",
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=DEFAULT_BUDGET_FRACTION,
        help=f"Fraction of total budget to spend (default {DEFAULT_BUDGET_FRACTION}).",
    )
    parser.add_argument(
        "--initial-trail-percent",
        type=float,
        default=DEFAULT_BUY_TRAIL_PERCENT,
        help=(
            "IB trailing percent points after fill "
            f"(default {DEFAULT_BUY_TRAIL_PERCENT} = {DEFAULT_BUY_TRAIL_PERCENT}%%)."
        ),
    )
    parser.add_argument(
        "--tight-trail-percent",
        type=float,
        default=DEFAULT_BUY_TRAIL_TIGHT_PERCENT,
        help=(
            "IB trailing percent at 1× target mid "
            f"(default {DEFAULT_BUY_TRAIL_TIGHT_PERCENT} = {DEFAULT_BUY_TRAIL_TIGHT_PERCENT}%%)."
        ),
    )
    parser.add_argument(
        "--trail-1-5x-percent",
        type=float,
        default=DEFAULT_BUY_TRAIL_1_5X_PERCENT,
        help=(
            "IB trailing percent at 1.5× target mid "
            f"(default {DEFAULT_BUY_TRAIL_1_5X_PERCENT} = {DEFAULT_BUY_TRAIL_1_5X_PERCENT}%%)."
        ),
    )
    parser.add_argument(
        "--trail-2x-percent",
        type=float,
        default=DEFAULT_BUY_TRAIL_2X_PERCENT,
        help=(
            "IB trailing percent at 2× target mid "
            f"(default {DEFAULT_BUY_TRAIL_2X_PERCENT} = {DEFAULT_BUY_TRAIL_2X_PERCENT}%%)."
        ),
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="Net profit fraction after fees for 1× tighten (default env / ~0.3836%%).",
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
        help="Live buy + live trail watcher (needs TRADINGAI_IBKR_ALLOW_ORDERS=1).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirm when using --submit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"Poll interval before first tighten (default {DEFAULT_POLL_INTERVAL_SEC:g}).",
    )
    parser.add_argument(
        "--tight-interval",
        type=float,
        default=DEFAULT_TIGHT_POLL_INTERVAL_SEC,
        help=(
            "Poll interval after 0.05%% trail is set "
            f"(default {DEFAULT_TIGHT_POLL_INTERVAL_SEC:g})."
        ),
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run fill/trail watcher in this process (internal worker mode).",
    )
    parser.add_argument("--job-id", default=None, help="Job id for --foreground.")
    parser.add_argument(
        "--buy-order-id",
        type=int,
        default=None,
        help="Buy order id (required with --foreground).",
    )
    parser.add_argument(
        "--target-qty",
        type=float,
        default=None,
        help="Expected buy quantity (required with --foreground).",
    )
    parser.add_argument(
        "--activate-price",
        type=float,
        default=None,
        help="Trail activate mid/limit (required with --foreground).",
    )
    parser.add_argument(
        "--buy-limit-price",
        type=float,
        default=None,
        help="Optional buy limit metadata for --foreground.",
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
        help="Stop a running buy-trail job.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.status is not None:
        print(json.dumps(job_status(args.status or None), indent=2, default=str))
        return
    if args.stop:
        print(json.dumps(stop_job(args.stop), indent=2, default=str))
        return

    if args.foreground:
        if (
            not args.symbol
            or args.buy_order_id is None
            or args.target_qty is None
            or args.activate_price is None
        ):
            parser.error(
                "--foreground requires symbol, --buy-order-id, "
                "--target-qty, and --activate-price"
            )
        if not args.job_id:
            parser.error("--foreground requires --job-id")
        config = BuyTrailConfig(
            symbol=args.symbol,
            buy_order_id=int(args.buy_order_id),
            target_qty=float(args.target_qty),
            activate_price=float(args.activate_price),
            initial_trail_percent=float(args.initial_trail_percent),
            tight_trail_percent=float(args.tight_trail_percent),
            trail_1_5x_percent=float(args.trail_1_5x_percent),
            trail_2x_percent=float(args.trail_2x_percent),
            target_return=args.target_return,
            tif=args.tif,
            outside_rth=args.outside_rth,
            account=args.account,
            submit=bool(args.submit),
            interval_sec=float(args.interval),
            tight_interval_sec=float(args.tight_interval),
            buy_limit_price=args.buy_limit_price,
        )
        print(
            json.dumps(
                {"job_id": args.job_id, "mode": "foreground", "config": asdict(config)},
                indent=2,
                default=str,
            ),
            flush=True,
        )
        run_watcher_loop(args.job_id, config)
        return

    if not args.symbol:
        parser.error("symbol is required unless using --status / --stop / --foreground")

    if args.submit and not args.yes:
        reply = input(
            f"Submit LIVE budget buy + trail watcher for {args.symbol.upper()}? [y/N] "
        ).strip().lower()
        if reply not in {"y", "yes"}:
            print(json.dumps({"cancelled": True}, indent=2))
            raise SystemExit(0)

    result = start_buy_trail(
        symbol=args.symbol,
        total_budget=args.total_budget,
        limit_price=args.limit_price,
        budget_fraction=args.budget_fraction,
        initial_trail_percent=args.initial_trail_percent,
        tight_trail_percent=args.tight_trail_percent,
        trail_1_5x_percent=args.trail_1_5x_percent,
        trail_2x_percent=args.trail_2x_percent,
        target_return=args.target_return,
        tif=args.tif,
        outside_rth=args.outside_rth,
        account=args.account,
        submit=bool(args.submit),
        interval_sec=float(args.interval),
        tight_interval_sec=float(args.tight_interval),
    )
    print(json.dumps(result, indent=2, default=str))
    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
