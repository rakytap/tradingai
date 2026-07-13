from __future__ import annotations

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.stocks import list_tickers
from tradingai.predict.tools.predict import generate_prediction_tool


def _resolve_tickers(tickers: list[str] | None) -> list[str]:
    if tickers:
        return [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    registered = list_tickers()
    if registered:
        return registered
    return list(get_sync_tickers())


def generate_predictions(tickers: list[str] | None = None) -> dict:
    initialize_schema()
    tickers = _resolve_tickers(tickers)
    generated = 0
    errors: list[str] = []
    summaries: list[dict] = []

    for ticker in tickers:
        try:
            print(f"Generating predictions for {ticker}")
            payload = generate_prediction_tool(ticker)
            generated += 1
            horizons = payload.get("horizons", [])
            primary = horizons[0] if horizons else {}
            summaries.append(
                {
                    "ticker": ticker,
                    "combined_score": payload.get("fusion", {}).get("combined_score"),
                    "probability_up_5d": next(
                        (
                            h.get("probability_up")
                            for h in horizons
                            if h.get("horizon_days") == 5
                        ),
                        primary.get("probability_up"),
                    ),
                    "prediction_ids": payload.get("prediction_ids", []),
                }
            )
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    return {
        "total": len(tickers),
        "generated": generated,
        "failed": len(errors),
        "errors": errors[:25],
        "predictions": summaries,
    }


def main() -> None:
    import json

    print(json.dumps(generate_predictions(), indent=2))


if __name__ == "__main__":
    main()
