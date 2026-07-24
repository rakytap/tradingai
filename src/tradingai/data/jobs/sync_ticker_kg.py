from __future__ import annotations

import argparse

from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.ticker_kg import upsert_ticker_kg_nodes
from tradingai.news.entity_graph import ENTITY_GRAPH


def sync_ticker_kg(*, preserve_overlays: bool = True) -> dict:
    initialize_schema()
    written = upsert_ticker_kg_nodes(preserve_overlays=preserve_overlays)
    return {
        "nodes_written": written,
        "preserve_overlays": preserve_overlays,
        "tickers": sorted(ENTITY_GRAPH.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsert curated ENTITY_GRAPH into ticker_kg_nodes."
    )
    parser.add_argument(
        "--preserve-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep non-empty DB relation arrays (LLM overlays). Default: true.",
    )
    parser.add_argument(
        "--force-seed",
        action="store_true",
        help="Replace DB rows fully from ENTITY_GRAPH (disables preserve-overlays).",
    )
    args = parser.parse_args()
    preserve = False if args.force_seed else bool(args.preserve_overlays)
    result = sync_ticker_kg(preserve_overlays=preserve)
    print(
        f"tradingai-sync-ticker-kg: wrote {result['nodes_written']} nodes "
        f"(preserve_overlays={result['preserve_overlays']})"
    )
    print(f"tickers: {', '.join(result['tickers'])}")


if __name__ == "__main__":
    main()
