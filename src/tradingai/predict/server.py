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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
