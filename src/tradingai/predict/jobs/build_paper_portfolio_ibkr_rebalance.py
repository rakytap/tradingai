from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingai.predict.portfolio import (
    DEFAULT_COMMISSION_PREFERENCE,
    DEFAULT_PREVIOUS_PORTFOLIO_PATH,
    build_paper_portfolio_ibkr_rebalance,
)
from tradingai.predict.portfolio_risk import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_AVERSION,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reoptimize a prior paper portfolio with soft IBKR Pro Tiered "
            "US-stock commission preference."
        )
    )
    parser.add_argument(
        "--previous-portfolio",
        type=Path,
        default=DEFAULT_PREVIOUS_PORTFOLIO_PATH,
        help=f"Prior portfolio JSON (default {DEFAULT_PREVIOUS_PORTFOLIO_PATH}).",
    )
    parser.add_argument(
        "--commission-preference",
        type=float,
        default=DEFAULT_COMMISSION_PREFERENCE,
        help=(
            "Soft IBKR cost weight gamma (default "
            f"{DEFAULT_COMMISSION_PREFERENCE}; larger = stickier to prior)."
        ),
    )
    parser.add_argument(
        "--monthly-volume-shares",
        type=float,
        default=0.0,
        help="Month-to-date share volume for Tiered rate selection (default 0).",
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
        help="Reference account size for target_usd / fees (default 100000).",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.55,
        help="Minimum probability_up to include (default 0.55).",
    )
    parser.add_argument(
        "--max-per-sector",
        type=int,
        default=4,
        help="Max holdings from one sector (default 4).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Daily return lookback for cov (default {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=DEFAULT_RISK_AVERSION,
        help=f"Markowitz risk aversion lambda (default {DEFAULT_RISK_AVERSION}).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=DEFAULT_MAX_WEIGHT,
        help=f"Max per-name weight (default {DEFAULT_MAX_WEIGHT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/paper_portfolio_ibkr_rebalance.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    portfolio = build_paper_portfolio_ibkr_rebalance(
        previous_portfolio_path=args.previous_portfolio,
        commission_preference=args.commission_preference,
        monthly_volume_shares=args.monthly_volume_shares,
        horizon_days=args.horizon,
        positions=args.positions,
        cash_weight=args.cash_weight,
        min_probability_up=args.min_probability,
        max_per_sector=args.max_per_sector,
        notional_usd=args.notional,
        lookback_days=args.lookback,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(portfolio, indent=2))

    csv_path = args.output.with_suffix(".csv")
    lines = [
        "ticker,weight_pct,prior_weight,target_usd,trade_usd,est_commission_usd,"
        "probability_up,combined_score,annualized_volatility_pct,sector"
    ]
    for holding in portfolio["holdings"]:
        lines.append(
            f"{holding['ticker']},{holding['weight_pct']},"
            f"{holding.get('prior_weight', '')},{holding.get('target_usd', '')},"
            f"{holding.get('trade_usd', '')},{holding.get('est_commission_usd', '')},"
            f"{holding['probability_up']},{holding['combined_score']},"
            f"{holding.get('annualized_volatility_pct', '')},"
            f"{holding.get('sector') or ''}"
        )
    csv_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(portfolio, indent=2))
    print(f"\nSaved to {args.output.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    print(
        "Estimated IBKR Tiered commission USD: "
        f"{portfolio.get('estimated_ibkr_commission_usd')}"
    )


if __name__ == "__main__":
    main()
