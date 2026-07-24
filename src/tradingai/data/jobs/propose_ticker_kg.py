from __future__ import annotations

import argparse
import json

from tradingai.news.kg_curator.llm_client import LLMCuratorError
from tradingai.news.kg_curator.propose import propose_ticker_graph_edges


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose L3 entity-graph edges via OpenAI-compatible LLM (human review)."
    )
    parser.add_argument("ticker", help="Subject ticker, e.g. MSFT")
    parser.add_argument(
        "--mode",
        choices=("merge", "replace"),
        default="merge",
        help="Merge with existing node (default) or replace relation lists.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "groq"),
        help="LLM provider (default: TRADINGAI_LLM_PROVIDER or openai).",
    )
    parser.add_argument(
        "--model",
        help="Override model (OPENAI_MODEL or GROQ_MODEL depending on provider).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Call LLM and print proposal without writing a file.",
    )
    args = parser.parse_args()
    try:
        proposal = propose_ticker_graph_edges(
            args.ticker,
            mode=args.mode,
            model=args.model,
            provider=args.provider,
            dry_run=args.dry_run,
        )
    except LLMCuratorError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(proposal, indent=2, default=str))
    if proposal.get("path"):
        print(f"\nReview then apply with: tradingai-apply-ticker-kg {proposal['path']}")


if __name__ == "__main__":
    main()
