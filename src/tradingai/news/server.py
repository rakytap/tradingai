from mcp.server.fastmcp import FastMCP

from tradingai.news.tools import news

mcp = FastMCP(
    "tradingai-news",
    instructions=(
        "Financial news and sentiment analysis tools. "
        "Outputs are informational only and not financial advice."
    ),
)


@mcp.tool()
def fetch_company_news(
    ticker: str,
    count: int = 10,
    rss_url: str | None = None,
) -> dict:
    """Fetch recent company news headlines from enabled providers.

    Args:
        ticker: Stock ticker symbol, e.g. NVDA.
        count: Maximum number of headlines to return.
        rss_url: Optional company RSS feed override.
    """
    return news.fetch_company_news(ticker=ticker, count=count, rss_url=rss_url)


@mcp.tool()
def fetch_press_releases(
    ticker: str,
    count: int = 10,
    rss_url: str | None = None,
) -> dict:
    """Fetch company press releases and IR announcements.

    Args:
        ticker: Stock ticker symbol.
        count: Maximum number of releases to return.
        rss_url: Optional company RSS feed override.
    """
    return news.fetch_press_releases(ticker=ticker, count=count, rss_url=rss_url)


@mcp.tool()
def fetch_sec_filings(
    ticker: str,
    forms: list[str] | None = None,
    count: int = 5,
) -> dict:
    """Fetch recent SEC filings such as 8-K, 10-Q, and 10-K.

    Args:
        ticker: Stock ticker symbol.
        forms: Optional SEC form types to filter.
        count: Maximum number of filings to return.
    """
    return news.fetch_sec_filings(ticker=ticker, forms=forms, count=count)


@mcp.tool()
def fetch_earnings_materials(ticker: str) -> dict:
    """Fetch the latest earnings press release and transcript when available.

    Args:
        ticker: Stock ticker symbol.
    """
    return news.fetch_earnings_materials(ticker=ticker)


@mcp.tool()
def fetch_analyst_actions(ticker: str, count: int = 10) -> dict:
    """Fetch analyst upgrades, downgrades, and recommendation changes.

    Args:
        ticker: Stock ticker symbol.
        count: Maximum number of analyst actions to return.
    """
    return news.fetch_analyst_actions(ticker=ticker, count=count)


@mcp.tool()
def fetch_social_sentiment(ticker: str, count: int = 15) -> dict:
    """Fetch filtered Reddit/X posts and a social-only sentiment score.

    Social sources are noise-filtered: low-effort posts, spam, and non-ticker
    mentions are removed. Social sentiment is weighted lower than news/filings.

    Args:
        ticker: Stock ticker symbol.
        count: Maximum number of social posts to return.
    """
    return news.fetch_social_sentiment(ticker=ticker, count=count)


@mcp.tool()
def analyze_news_sentiment(
    ticker: str,
    count: int = 10,
    rss_url: str | None = None,
) -> dict:
    """Analyze news, filings, analyst actions, and social sentiment for a report.

    Returns structured sentiment, catalysts, risk factors, and confidence.
    Includes separate social_sentiment from Reddit/X with noise filtering.
    This is informational analysis, not financial advice.

    Args:
        ticker: Stock ticker symbol.
        count: Maximum number of news items per provider.
        rss_url: Optional company RSS feed override.
    """
    return news.analyze_news_sentiment(ticker=ticker, count=count, rss_url=rss_url)


@mcp.tool()
def get_news_events(ticker: str, limit: int = 20) -> dict:
    """Inspect L2 typed news-graph events and sector exposures for a ticker.

    Extracts deterministic event types (e.g. ai_capex_increase, earnings_beat)
    from headlines/filings, propagates themes through the static ontology to
    sectors, and returns live + stored events. Sentiment remains debug metadata;
    fusion news score uses event exposure when events match.

    Args:
        ticker: Stock ticker symbol, e.g. NVDA.
        limit: Max stored graph events to return from the database.
    """
    return news.get_news_events(ticker=ticker, limit=limit)


@mcp.tool()
def get_ticker_graph(ticker: str) -> dict:
    """Inspect the L3 ticker knowledge-graph node and one-hop neighbors.

    Resolves DB overlay (LLM-applied) over the static entity_graph.py seed.
    Peers, suppliers, customers, thematic proxies, commodities, rate/FX flags.

    Args:
        ticker: Stock ticker symbol, e.g. MSFT.
    """
    return news.get_ticker_graph(ticker=ticker)


@mcp.tool()
def get_related_tickers(ticker: str, limit: int = 12) -> dict:
    """Expand L2 news events through the L3 graph into related ticker candidates.

    Example: MSFT AI capex → SMH / utilities / grid names. Informational only;
    does not change the subject's fusion news directional_score.

    Args:
        ticker: Stock ticker symbol, e.g. MSFT.
        limit: Maximum related candidates to return.
    """
    return news.get_related_tickers(ticker=ticker, limit=limit)


@mcp.tool()
def propose_ticker_graph_edges(
    ticker: str,
    mode: str = "merge",
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    """LLM proposes L3 peers/suppliers/customers/thematic_proxies for review.

    Writes a pending JSON under TRADINGAI_KG_PROPOSAL_DIR. Does not change live
    scoring until apply_ticker_graph_edges is called with that path.
    Providers: openai (OPENAI_API_KEY) or groq (GROQ_API_KEY). Set
    TRADINGAI_LLM_PROVIDER or pass provider=.

    Args:
        ticker: Subject ticker, e.g. MSFT.
        mode: merge (default) or replace relation lists.
        model: Optional model override (OPENAI_MODEL / GROQ_MODEL).
        provider: Optional openai or groq override.
    """
    return news.propose_ticker_graph_edges(
        ticker=ticker,
        mode=mode,
        model=model,
        provider=provider,
    )


@mcp.tool()
def apply_ticker_graph_edges(proposal_path: str) -> dict:
    """Apply a reviewed proposal JSON into ticker_kg_nodes (DB overlay).

    Explicit path required — no auto-apply. Runtime related_candidates then
    prefer the DB overlay over the static ENTITY_GRAPH seed.

    Args:
        proposal_path: Path returned by propose_ticker_graph_edges.
    """
    return news.apply_ticker_graph_edges(proposal_path=proposal_path)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
