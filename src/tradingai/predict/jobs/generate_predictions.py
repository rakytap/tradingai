from __future__ import annotations

import argparse

from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.etf_symbols import is_etf
from tradingai.data.repositories.stocks import get_stock, list_tickers
from tradingai.predict.tools.predict import generate_prediction_tool
from tradingai.regime.sector_proxies import normalize_sector_label
from tradingai.regime.tools.regime import load_or_build_regime_state


def _resolve_tickers(tickers: list[str] | None) -> list[str]:
    if tickers:
        return [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    registered = list_tickers()
    if registered:
        return registered
    return list(get_sync_tickers())


def _filter_bullish_sectors(tickers: list[str]) -> list[str]:
    state = load_or_build_regime_state()
    bullish = {
        item.sector
        for item in state.sectors
        if str(item.stance).lower() == "bullish"
    }
    kept: list[str] = []
    for ticker in tickers:
        if is_etf(ticker):
            continue
        try:
            stock = get_stock(ticker)
        except Exception:
            stock = None
        sector = normalize_sector_label(stock.get("sector") if stock else None)
        if sector in bullish:
            kept.append(ticker)
    return kept


def generate_predictions(
    tickers: list[str] | None = None,
    *,
    bullish_sectors_only: bool = False,
) -> dict:
    initialize_schema()
    tickers = _resolve_tickers(tickers)
    if bullish_sectors_only:
        tickers = _filter_bullish_sectors(tickers)
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
        "bullish_sectors_only": bullish_sectors_only,
    }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Batch-generate fused predictions.")
    parser.add_argument("--tickers", nargs="*", help="Optional ticker subset.")
    parser.add_argument(
        "--bullish-sectors-only",
        action="store_true",
        help="Only tickers in L4 bullish sectors (L5 ranking universe).",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            generate_predictions(
                args.tickers,
                bullish_sectors_only=args.bullish_sectors_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
