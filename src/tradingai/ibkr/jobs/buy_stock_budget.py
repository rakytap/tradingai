"""CLI: budget-sized IBKR LIMIT buy (integer shares, dry-run by default)."""

from __future__ import annotations

import argparse
import json
import sys

from tradingai.ibkr.tools.orders import buy_stock_budget_tool
from tradingai.intraday.config import get_buy_notional_usd
from tradingai.providers.ibkr_tws import (
    DEFAULT_BUDGET_FRACTION,
    DEFAULT_BUY_STOP_LOSS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Place a LIMIT buy via IBKR TWS sized to a cash budget. "
            "Spends budget_fraction (default 0.95) of total_budget on "
            "floor(spend / limit_price) whole shares. "
            "Default limit_price is current mid. Attaches a SELL STP at "
            f"{DEFAULT_BUY_STOP_LOSS:.0%} under the limit unless --no-stop-loss. "
            "Dry-run unless --submit."
        )
    )
    parser.add_argument("symbol", help="Ticker, e.g. COIN.")
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
        help="Limit price (default: current IB mid).",
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=DEFAULT_BUDGET_FRACTION,
        help=f"Fraction of total budget to spend (default {DEFAULT_BUDGET_FRACTION}).",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULT_BUY_STOP_LOSS,
        help=(
            f"Stop-loss fraction under the buy limit "
            f"(default {DEFAULT_BUY_STOP_LOSS} = {DEFAULT_BUY_STOP_LOSS:.0%}). "
            "Use 0 with --no-stop-loss to disable."
        ),
    )
    parser.add_argument(
        "--no-stop-loss",
        action="store_true",
        help="Do not attach a SELL STP child order.",
    )
    parser.add_argument(
        "--tif",
        default="DAY",
        help="Time in force (default DAY).",
    )
    parser.add_argument(
        "--outside-rth",
        action="store_true",
        help="Allow outside regular trading hours.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="Optional IB account code.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually send the order (requires TRADINGAI_IBKR_ALLOW_ORDERS=1). "
            "Without this flag the order is dry-run only."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirm when using --submit.",
    )
    args = parser.parse_args()
    stop_loss = None if args.no_stop_loss else float(args.stop_loss)

    dry_run = not args.submit
    if args.submit and not args.yes:
        preview = buy_stock_budget_tool(
            symbol=args.symbol,
            total_budget=args.total_budget,
            limit_price=args.limit_price,
            budget_fraction=args.budget_fraction,
            stop_loss=stop_loss,
            tif=args.tif,
            outside_rth=args.outside_rth,
            account=args.account,
            dry_run=True,
        )
        print(json.dumps(preview, indent=2, default=str))
        if preview.get("error") or preview.get("status") == "rejected":
            print("Preview failed; not submitting.", file=sys.stderr)
            raise SystemExit(1)
        sizing = preview.get("sizing") or preview
        qty = sizing.get("quantity") or preview.get("quantity")
        px = sizing.get("limit_price") or preview.get("limit_price")
        spend = sizing.get("spend_budget")
        stop_px = sizing.get("stop_price") or preview.get("stop_price")
        stop_bit = f", stop@{stop_px}" if stop_px is not None else ", no stop"
        reply = input(
            f"Submit LIVE LIMIT buy {args.symbol.upper()} "
            f"qty={qty} @ {px} (spend≈{spend}{stop_bit})? [y/N] "
        ).strip().lower()
        if reply not in {"y", "yes"}:
            print(json.dumps({"cancelled": True, "dry_run_preview": preview}, indent=2, default=str))
            raise SystemExit(0)

    result = buy_stock_budget_tool(
        symbol=args.symbol,
        total_budget=args.total_budget,
        limit_price=args.limit_price,
        budget_fraction=args.budget_fraction,
        stop_loss=stop_loss,
        tif=args.tif,
        outside_rth=args.outside_rth,
        account=args.account,
        dry_run=dry_run,
    )
    print(json.dumps(result, indent=2, default=str))

    if result.get("error") or result.get("status") == "rejected":
        raise SystemExit(1)
    if args.submit and not result.get("submitted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
