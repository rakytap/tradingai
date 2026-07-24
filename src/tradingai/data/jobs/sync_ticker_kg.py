from __future__ import annotations

from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.ticker_kg import upsert_ticker_kg_nodes
from tradingai.news.entity_graph import ENTITY_GRAPH


def sync_ticker_kg() -> dict:
    initialize_schema()
    written = upsert_ticker_kg_nodes()
    return {
        "nodes_written": written,
        "tickers": sorted(ENTITY_GRAPH.keys()),
    }


def main() -> None:
    result = sync_ticker_kg()
    print(f"tradingai-sync-ticker-kg: wrote {result['nodes_written']} nodes")
    print(f"tickers: {', '.join(result['tickers'])}")


if __name__ == "__main__":
    main()
