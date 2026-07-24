from __future__ import annotations

from tradingai.analysis.ml.config import get_default_ml_model, get_ml_label_mode
from tradingai.analysis.ml.inference import predict_direction
from tradingai.data.repositories.stocks import list_tickers_by_sector
from tradingai.predict.config import get_ml_horizon_days, get_ml_model_name
from tradingai.regime.sector_proxies import normalize_sector_label
from tradingai.regime.tools.regime import load_or_build_regime_state


def predict_cs_score(
    ticker: str,
    *,
    model_name: str | None = None,
    horizon_days: int | None = None,
) -> dict:
    payload = predict_direction(
        ticker,
        model_name=model_name or get_ml_model_name() or get_default_ml_model(),
        horizon_days=horizon_days or get_ml_horizon_days(),
    )
    cs_prob = payload.get("cs_prob")
    if cs_prob is None:
        cs_prob = payload.get("probability_up")
    return {
        **payload,
        "ml_score": float(cs_prob),
        "cs_prob": float(cs_prob),
    }


def rank_tickers_in_sector(
    sector: str,
    *,
    horizon_days: int | None = None,
    model_name: str | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    canonical = normalize_sector_label(sector) or sector
    universe = tickers or list_tickers_by_sector(canonical, exclude_etf=True)
    horizon = horizon_days or get_ml_horizon_days()
    model = model_name or get_ml_model_name()

    scored: list[dict] = []
    errors: list[str] = []
    for ticker in universe:
        try:
            payload = predict_cs_score(
                ticker,
                model_name=model,
                horizon_days=horizon,
            )
            scored.append(
                {
                    "ticker": payload["ticker"],
                    "sector": canonical,
                    "ml_score": payload["ml_score"],
                    "cs_prob": payload["cs_prob"],
                    "label_mode": payload.get("label_mode") or get_ml_label_mode(),
                    "model": payload.get("model"),
                    "model_source": payload.get("model_source"),
                    "horizon_days": horizon,
                }
            )
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    ranked = sorted(scored, key=lambda item: (-float(item["ml_score"]), item["ticker"]))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        item["sector_rank"] = index
    if limit is not None:
        ranked = ranked[:limit]
    if errors:
        for item in ranked:
            item.setdefault("errors_omitted", len(errors))
    return ranked


def bullish_sectors_from_regime() -> list[str]:
    state = load_or_build_regime_state()
    return [
        item.sector
        for item in state.sectors
        if str(item.stance).lower() == "bullish"
    ]


def rank_bullish_sector_candidates(
    *,
    sector: str | None = None,
    limit: int = 10,
    per_sector_limit: int | None = None,
    horizon_days: int | None = None,
    model_name: str | None = None,
) -> dict:
    if sector:
        sectors = [normalize_sector_label(sector) or sector]
    else:
        sectors = bullish_sectors_from_regime()

    per_limit = per_sector_limit if per_sector_limit is not None else limit
    by_sector: dict[str, list[dict]] = {}
    merged: list[dict] = []
    for name in sectors:
        ranked = rank_tickers_in_sector(
            name,
            horizon_days=horizon_days,
            model_name=model_name,
            limit=per_limit,
        )
        by_sector[name] = ranked
        merged.extend(ranked)

    merged_sorted = sorted(merged, key=lambda item: (-float(item["ml_score"]), item["ticker"]))
    for index, item in enumerate(merged_sorted, start=1):
        item["global_rank"] = index
    return {
        "sectors": sectors,
        "label_mode": get_ml_label_mode(),
        "by_sector": by_sector,
        "candidates": merged_sorted[:limit],
    }
