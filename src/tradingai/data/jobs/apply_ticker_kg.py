from __future__ import annotations

import argparse
import json

from tradingai.news.kg_curator.apply import apply_proposal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a reviewed L3 entity-graph proposal into ticker_kg_nodes."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to proposal JSON from tradingai-propose-ticker-kg",
    )
    parser.add_argument(
        "--ticker",
        help="Apply latest pending proposal for this ticker (if path omitted).",
    )
    args = parser.parse_args()
    if not args.path and not args.ticker:
        raise SystemExit("Provide proposal path or --ticker")
    result = apply_proposal(args.path, ticker=args.ticker)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
