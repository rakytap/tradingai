from mcp.server.fastmcp import FastMCP

from tradingai.predict.tools import predict

mcp = FastMCP(
    "tradingai-predict",
    instructions=(
        "Predictive signal fusion tools that combine technical, news, and regime "
        "signals into deterministic multi-horizon probability estimates. "
        "Probabilities are computed by a fusion function, not by an LLM. "
        "Outputs are informational only and not financial advice."
    ),
)


@mcp.tool()
def get_agent_signals(ticker: str) -> dict:
    """Return normalized per-agent signals for technical, news, and regime inputs."""
    return predict.get_agent_signals_tool(ticker)


@mcp.tool()
def generate_prediction(ticker: str) -> dict:
    """Generate a multi-horizon prediction and persist it to PostgreSQL.

    Probabilities are computed by a deterministic weighted-sigmoid fusion function,
    not by an LLM.
    """
    return predict.generate_prediction_tool(ticker)


@mcp.tool()
def explain_prediction(ticker: str) -> dict:
    """Return a human-readable explanation with deterministic probability numbers."""
    return predict.explain_prediction_tool(ticker)


@mcp.tool()
def evaluate_predictions(horizon_days: int | None = None) -> dict:
    """Backfill actual returns for mature predictions and update outcomes."""
    return predict.evaluate_predictions_tool(horizon_days)


@mcp.tool()
def get_evaluation_metrics(horizon_days: int | None = None) -> dict:
    """Return accuracy, precision/recall, MAE, Brier score, and top-10 ranked return."""
    return predict.get_evaluation_metrics_tool(horizon_days)


@mcp.tool()
def get_prediction_history(ticker: str | None = None, limit: int = 20) -> dict:
    """Query stored predictions for a ticker or across all tickers."""
    return predict.get_prediction_history_tool(ticker=ticker, limit=limit)


@mcp.tool()
def get_paper_portfolio(
    positions: int = 10,
    cash_weight: float = 0.10,
    min_probability_up: float = 0.55,
    max_per_sector: int = 4,
    notional_usd: float = 100_000.0,
    horizon_days: int | None = None,
    optimizer: str = "score",
    lookback_days: int = 126,
    risk_aversion: float = 3.0,
    max_weight: float = 0.25,
) -> dict:
    """Build a diversified paper-trading portfolio from latest ranked predictions.

    optimizer='score' weights by combined_score (default).
    optimizer='markowitz' uses prediction expected returns plus covariance from
    synced daily_prices (requires tradingai-sync-prices).
    max_per_sector caps names from any one sector (default 4).
    """
    return predict.get_paper_portfolio_tool(
        positions=positions,
        cash_weight=cash_weight,
        min_probability_up=min_probability_up,
        max_per_sector=max_per_sector,
        notional_usd=notional_usd,
        horizon_days=horizon_days,
        optimizer=optimizer,
        lookback_days=lookback_days,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )


@mcp.tool()
def get_paper_portfolio_ibkr_rebalance(
    previous_portfolio_path: str = "data/paper_portfolio.json",
    commission_preference: float = 1.0,
    monthly_volume_shares: float = 0.0,
    positions: int = 10,
    cash_weight: float = 0.10,
    min_probability_up: float = 0.55,
    max_per_sector: int = 4,
    notional_usd: float = 100_000.0,
    horizon_days: int | None = None,
    lookback_days: int = 126,
    risk_aversion: float = 3.0,
    max_weight: float = 0.25,
) -> dict:
    """Reoptimize a prior paper portfolio with soft IBKR Pro Tiered US commissions.

    Requires a previous portfolio JSON (default data/paper_portfolio.json).
    commission_preference (gamma) weights estimated Tiered $/share turnover vs
    mean-variance; larger values keep the book closer to the prior.
    monthly_volume_shares selects the Tiered marginal rate. Fee estimate is
    broker commission only (no exchange/regulatory pass-through).
    """
    return predict.get_paper_portfolio_ibkr_rebalance_tool(
        previous_portfolio_path=previous_portfolio_path,
        commission_preference=commission_preference,
        monthly_volume_shares=monthly_volume_shares,
        positions=positions,
        cash_weight=cash_weight,
        min_probability_up=min_probability_up,
        max_per_sector=max_per_sector,
        notional_usd=notional_usd,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )


@mcp.tool()
def get_prediction_track_record(
    sector: str | None = None,
    prediction_type: str | None = None,
    catalyst_tag: str | None = None,
    regime: str | None = None,
    horizon_days: int | None = None,
) -> dict:
    """Return segment hit rate, average return, and overconfidence from stored history."""
    return predict.get_prediction_track_record_tool(
        sector=sector,
        prediction_type=prediction_type,
        catalyst_tag=catalyst_tag,
        regime=regime,
        horizon_days=horizon_days,
    )


@mcp.tool()
def get_segment_calibration(horizon_days: int | None = None) -> dict:
    """Return calibration metrics grouped by sector and prediction type."""
    return predict.get_segment_calibration_tool(horizon_days)


@mcp.tool()
def rank_sector_candidates(
    sector: str | None = None,
    limit: int = 10,
    horizon_days: int | None = None,
) -> dict:
    """L5 cross-sectional ranks inside L4 bullish sectors (or one sector).

    ML scores who beats sector peers; L4 decides which sectors are eligible.
    When sector is omitted, unions top names across all bullish board sectors.
    """
    return predict.rank_sector_candidates_tool(
        sector=sector,
        limit=limit,
        horizon_days=horizon_days,
    )


@mcp.tool()
def scan_open_dip_bounces(
    tickers: list[str] | None = None,
    dip_pct: float | None = None,
    window_minutes: int | None = None,
    prefer_db: bool = True,
    cap: int | None = None,
    buy_notional_usd: float | None = None,
    target_return: float | None = None,
    sell_trail: float | None = None,
    profit_taker_buffer: float | None = None,
    session_day: str | None = None,
) -> dict:
    """L0 intraday scan: names that plunged after the open and may be bouncing.

    Uses 1m bars from DB (tradingai-sync-intraday / IBKR TWS) when prefer_db; else live yfinance.
    Universe defaults to L4 bullish-sector equities (capped). Each candidate includes
    profit_target with: profit_taker_price (nets target_return + 0.02% after IBKR
    Tiered buy+sell fees), trail_activate_price (fee-aware target_return/2), and
    sell_trail_pct (default target_return/2). session_day is YYYY-MM-DD (default
    today ET). Alert/scan only — not financial advice.
    """
    return predict.scan_open_dip_bounces_tool(
        tickers=tickers,
        dip_pct=dip_pct,
        window_minutes=window_minutes,
        prefer_db=prefer_db,
        cap=cap,
        buy_notional_usd=buy_notional_usd,
        target_return=target_return,
        sell_trail=sell_trail,
        profit_taker_buffer=profit_taker_buffer,
        session_day=session_day,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
