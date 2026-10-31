"""CLI: IBKR LIMIT sell (full position / fee-aware target by default)."""

from __future__ import annotations

import argparse
import json
import sys

from tradingai.ibkr.tools.orders import sell_stock_limit_tool


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Place a LIMIT sell via IBKR TWS. "
            "Default quantity = full open long position. "
            "Default limit_price = fee-aware target at "
            "TRADINGAI_OPEN_DIP_TARGET_RETURN on last buy / avg cost. "
            "Dry-run unless --submit."
        )
    )
    parser.add_argument("symbol", help="Ticker, e.g. COIN.")
    parser.add_argument(
        "--qty",
        "--quantity",
        dest="quantity",
        type=float,
        default=None,
        help="Shares to sell (default: all open long shares).",
    )
    parser.add_argument(
        "--price",
        "--limit-price",
        dest="limit_price",
        type=float,
        default=None,
        help=(
            "Limit price (default: fee-aware target on last buy / avg cost)."
        ),
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help=(
            "Net profit as a fraction after IBKR fees "
            "(e.g. 0.01 = 1%%). Used when --price is omitted. "
            "Default: TRADINGAI_OPEN_DIP_TARGET_RETURN / ~0.3836%%."
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

    dry_run = not args.submit
    if args.submit and not args.yes:
        preview = sell_stock_limit_tool(
            symbol=args.symbol,
            quantity=args.quantity,
            limit_price=args.limit_price,
            target_return=args.target_return,
            tif=args.tif,
            outside_rth=args.outside_rth,
            account=args.account,
            dry_run=True,
        )
        print(json.dumps(preview, indent=2, default=str))
        if preview.get("error") or preview.get("status") == "rejected":
            print("Preview failed; not submitting.", file=sys.stderr)
            raise SystemExit(1)
        qty = preview.get("quantity")
        px = preview.get("limit_price")
        ret = (preview.get("sizing") or {}).get("target_return") or preview.get(
            "target_return"
        )
        ret_bit = f" net≈{ret:.4%}" if isinstance(ret, (int, float)) else ""
        reply = input(
            f"Submit LIVE LIMIT sell {args.symbol.upper()} "
            f"qty={qty} @ {px}{ret_bit}? [y/N] "
        ).strip().lower()
        if reply not in {"y", "yes"}:
            print(
                json.dumps(
                    {"cancelled": True, "dry_run_preview": preview},
                    indent=2,
                    default=str,
                )
            )
            raise SystemExit(0)

    result = sell_stock_limit_tool(
        symbol=args.symbol,
        quantity=args.quantity,
        limit_price=args.limit_price,
        target_return=args.target_return,
        tif=args.tif,
        outside_rth=args.outside_rth,
        account=args.account,
        dry_run=dry_run,
    )
    print(json.dumps(result, indent=2, default=str))

    if result.get("error") or result.get("status") == "rejected":
        raise SystemExit(1)
    if args.submit and result.get("status") in {
        "Inactive",
        "Cancelled",
        "ApiCancelled",
        "Rejected",
    }:
        raise SystemExit(1)
    if args.submit and not result.get("submitted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
