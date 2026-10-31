"""CLI: sell-trail-watch — activate at 0.2% profit, step trail, then exit.

One-shot TRAIL LIMIT (no watcher) is ``tradingai-sell-trail``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from tradingai.ibkr.sell_trail import (
    DEFAULT_POLL_INTERVAL_SEC,
    SellTrailConfig,
    job_status,
    run_watcher_loop,
    start_sell_trail,
    stop_job,
)
from tradingai.providers.ibkr_tws import (
    DEFAULT_SELL_TRAIL_1_5X_PERCENT,
    DEFAULT_SELL_TRAIL_2X_PERCENT,
    DEFAULT_SELL_TRAIL_ACTIVATE_RETURN,
    DEFAULT_SELL_TRAIL_PERCENT,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch an open long: query IBKR buy price, wait until fee-aware "
            f"{DEFAULT_SELL_TRAIL_ACTIVATE_RETURN * 100:g}% profit, place a "
            f"{DEFAULT_SELL_TRAIL_PERCENT}% TRAIL LIMIT, tighten to "
            f"{DEFAULT_SELL_TRAIL_1_5X_PERCENT}% / "
            f"{DEFAULT_SELL_TRAIL_2X_PERCENT}% at 1.5× / 2× activation profit "
            f"(poll every {DEFAULT_POLL_INTERVAL_SEC:g}s). Exits when the "
            "position is flat or the final trail tighten is applied."
        )
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Ticker, e.g. COIN (required unless --status/--stop).",
    )
    parser.add_argument(
        "--qty",
        "--quantity",
        dest="quantity",
        type=float,
        default=None,
        help="Shares to trail (default: full open long).",
    )
    parser.add_argument(
        "--buy-price",
        type=float,
        default=None,
        help="Override buy reference (default: last IB buy fill / avg cost).",
    )
    parser.add_argument(
        "--activate-return",
        type=float,
        default=DEFAULT_SELL_TRAIL_ACTIVATE_RETURN,
        help=(
            "Fee-aware net return that activates the trail "
            f"(default {DEFAULT_SELL_TRAIL_ACTIVATE_RETURN})."
        ),
    )
    parser.add_argument(
        "--initial-trail-percent",
        type=float,
        default=DEFAULT_SELL_TRAIL_PERCENT,
        help=(
            "IB trailing percent at activation "
            f"(default {DEFAULT_SELL_TRAIL_PERCENT})."
        ),
    )
    parser.add_argument(
        "--trail-1-5x-percent",
        type=float,
        default=DEFAULT_SELL_TRAIL_1_5X_PERCENT,
        help=(
            "IB trailing percent at 1.5× activation profit "
            f"(default {DEFAULT_SELL_TRAIL_1_5X_PERCENT})."
        ),
    )
    parser.add_argument(
        "--trail-2x-percent",
        type=float,
        default=DEFAULT_SELL_TRAIL_2X_PERCENT,
        help=(
            "IB trailing percent at 2× activation profit "
            f"(default {DEFAULT_SELL_TRAIL_2X_PERCENT})."
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
        help="Live watcher (needs TRADINGAI_IBKR_ALLOW_ORDERS=1).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation when --submit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"Poll interval seconds (default {DEFAULT_POLL_INTERVAL_SEC:g}).",
    )
    parser.add_argument(
        "--status",
        nargs="?",
        const="",
        default=None,
        help="Show all jobs, or one job when given a job_id.",
    )
    parser.add_argument("--stop", default=None, help="Stop watcher by job_id.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --stop, send SIGKILL if needed.",
    )
    # Foreground watcher args (spawned by start_background_watcher).
    parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--buy-price-source", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.status is not None:
        jid = args.status.strip() or None
        print(json.dumps(job_status(jid), indent=2, default=str))
        return 0

    if args.stop:
        print(
            json.dumps(
                stop_job(args.stop, force=bool(args.force)),
                indent=2,
                default=str,
            )
        )
        return 0

    if not args.symbol:
        print("symbol is required unless --status/--stop", file=sys.stderr)
        return 2

    if args.foreground:
        if (
            not args.symbol
            or not args.job_id
            or args.buy_price is None
            or args.quantity is None
        ):
            print(
                "--foreground requires symbol, --job-id, --buy-price, --quantity",
                file=sys.stderr,
            )
            return 2
        config = SellTrailConfig(
            symbol=args.symbol,
            quantity=float(args.quantity),
            buy_price=float(args.buy_price),
            activate_return=float(args.activate_return),
            initial_trail_percent=float(args.initial_trail_percent),
            trail_1_5x_percent=float(args.trail_1_5x_percent),
            trail_2x_percent=float(args.trail_2x_percent),
            tif=args.tif,
            outside_rth=bool(args.outside_rth),
            account=args.account,
            submit=bool(args.submit),
            interval_sec=float(args.interval),
            buy_price_source=args.buy_price_source,
        )
        print(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "mode": "foreground",
                    "config": asdict(config),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        run_watcher_loop(args.job_id, config)
        return 0

    if args.submit and not args.yes:
        reply = input(
            f"Start LIVE sell-trail-watch for {args.symbol.upper()}? [y/N] "
        ).strip().lower()
        if reply not in {"y", "yes"}:
            print(json.dumps({"cancelled": True}, indent=2))
            return 1

    try:
        result = start_sell_trail(
            symbol=args.symbol,
            quantity=args.quantity,
            buy_price=args.buy_price,
            activate_return=float(args.activate_return),
            initial_trail_percent=float(args.initial_trail_percent),
            trail_1_5x_percent=float(args.trail_1_5x_percent),
            trail_2x_percent=float(args.trail_2x_percent),
            tif=args.tif,
            outside_rth=bool(args.outside_rth),
            account=args.account,
            submit=bool(args.submit),
            interval_sec=float(args.interval),
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc), "symbol": args.symbol}, indent=2))
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
