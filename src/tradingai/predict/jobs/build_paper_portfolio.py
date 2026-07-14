from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingai.predict.portfolio import build_paper_portfolio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a paper-trading portfolio from latest stored predictions."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Horizon in trading days (default: TRADINGAI_ML_HORIZON).",
    )
    parser.add_argument("--positions", type=int, default=10, help="Number of holdings.")
    parser.add_argument(
        "--cash-weight",
        type=float,
        default=0.10,
        help="Cash buffer as fraction of portfolio (default 0.10).",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=100_000.0,
        help="Reference account size for target_usd (default 100000).",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.55,
        help="Minimum probability_up to include (default 0.55).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/paper_portfolio.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    portfolio = build_paper_portfolio(
        horizon_days=args.horizon,
        positions=args.positions,
        cash_weight=args.cash_weight,
        min_probability_up=args.min_probability,
        notional_usd=args.notional,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(portfolio, indent=2))

    csv_path = args.output.with_suffix(".csv")
    lines = ["ticker,weight_pct,target_usd,probability_up,combined_score,sector"]
    for holding in portfolio["holdings"]:
        lines.append(
            f"{holding['ticker']},{holding['weight_pct']},{holding.get('target_usd', '')},"
            f"{holding['probability_up']},{holding['combined_score']},"
            f"{holding.get('sector') or ''}"
        )
    csv_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(portfolio, indent=2))
    print(f"\nSaved to {args.output.resolve()}")
    print(f"CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
