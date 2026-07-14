from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from tradingai.data.connection import get_connection
from tradingai.data.etf_symbols import is_etf
from tradingai.predict.config import get_ml_horizon_days

BROAD_MARKET_ETFS = frozenset({"SPY", "VOO", "IVV", "VTI", "QQQ", "DIA", "IWM", "RSP"})
DEFAULT_POSITIONS = 10
DEFAULT_CASH_WEIGHT = 0.10
DEFAULT_MIN_COMBINED_SCORE = 0.0
DEFAULT_MIN_PROBABILITY_UP = 0.55
DEFAULT_MAX_PER_SECTOR = 2
DEFAULT_MAX_BROAD_ETF = 1


def _latest_predictions_per_ticker(
    *,
    horizon_days: int,
    database_url: str | None = None,
) -> list[dict]:
    with get_connection(database_url) as conn:
        cur = conn.execute(
            """
            SELECT DISTINCT ON (ticker)
                   id, ticker, timestamp, horizon_days, probability_up,
                   expected_return_pct, confidence, combined_score,
                   sector, regime, prediction_type,
                   technical_score, news_score, regime_score,
                   model_name, actual_return_pct, direction_correct
            FROM predictions
            WHERE horizon_days = %s
            ORDER BY ticker, timestamp DESC
            """,
            (horizon_days,),
        )
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def _sector_key(row: dict) -> str | None:
    sector = row.get("sector")
    if sector is None or str(sector).strip() == "":
        return None
    return str(sector).strip()


def select_portfolio_positions(
    candidates: list[dict],
    *,
    size: int = DEFAULT_POSITIONS,
    min_combined_score: float = DEFAULT_MIN_COMBINED_SCORE,
    min_probability_up: float = DEFAULT_MIN_PROBABILITY_UP,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    max_broad_etf: int = DEFAULT_MAX_BROAD_ETF,
) -> list[dict]:
    filtered = [
        row
        for row in candidates
        if float(row["combined_score"]) > min_combined_score
        and float(row["probability_up"]) >= min_probability_up
    ]
    ranked = sorted(filtered, key=lambda row: float(row["combined_score"]), reverse=True)

    stocks = [row for row in ranked if not is_etf(str(row["ticker"]))]
    etfs = [row for row in ranked if is_etf(str(row["ticker"]))]

    selected: list[dict] = []
    sector_counts: dict[str, int] = defaultdict(int)
    broad_etf_count = 0

    def _can_add(row: dict) -> bool:
        ticker = str(row["ticker"]).upper()
        sector = _sector_key(row)
        if sector is not None and sector_counts[sector] >= max_per_sector:
            return False
        if ticker in BROAD_MARKET_ETFS and broad_etf_count >= max_broad_etf:
            return False
        return True

    def _add(row: dict) -> None:
        nonlocal broad_etf_count
        ticker = str(row["ticker"]).upper()
        sector = _sector_key(row)
        selected.append(row)
        if sector is not None:
            sector_counts[sector] += 1
        if ticker in BROAD_MARKET_ETFS:
            broad_etf_count += 1

    min_stocks = min(size // 2, len(stocks))
    for row in stocks:
        if len(selected) >= size:
            break
        if sum(1 for item in selected if not is_etf(str(item["ticker"]))) >= min_stocks:
            break
        if _can_add(row):
            _add(row)

    for pool in (stocks, etfs):
        for row in pool:
            if len(selected) >= size:
                break
            if row in selected:
                continue
            if _can_add(row):
                _add(row)

    return selected


def _score_weights(rows: list[dict], *, invested_fraction: float) -> list[dict]:
    if not rows:
        return []

    scores = [max(float(row["combined_score"]), 0.01) for row in rows]
    total = sum(scores)
    positions = []
    for row, score in zip(rows, scores):
        weight = invested_fraction * (score / total)
        positions.append(
            {
                "ticker": str(row["ticker"]).upper(),
                "weight": round(weight, 4),
                "weight_pct": round(weight * 100, 2),
                "combined_score": round(float(row["combined_score"]), 4),
                "probability_up": round(float(row["probability_up"]), 4),
                "expected_return_pct": round(float(row["expected_return_pct"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "sector": row.get("sector"),
                "regime": row.get("regime"),
                "prediction_type": row.get("prediction_type"),
                "technical_score": round(float(row["technical_score"]), 4),
                "news_score": round(float(row["news_score"]), 4),
                "regime_score": round(float(row["regime_score"]), 4),
                "model_name": row.get("model_name"),
                "prediction_id": row.get("id"),
                "as_of": row.get("timestamp").isoformat() if row.get("timestamp") else None,
                "is_etf": is_etf(str(row["ticker"])),
            }
        )
    return positions


def build_paper_portfolio(
    *,
    horizon_days: int | None = None,
    positions: int = DEFAULT_POSITIONS,
    cash_weight: float = DEFAULT_CASH_WEIGHT,
    min_combined_score: float = DEFAULT_MIN_COMBINED_SCORE,
    min_probability_up: float = DEFAULT_MIN_PROBABILITY_UP,
    notional_usd: float | None = 100_000.0,
    database_url: str | None = None,
) -> dict:
    horizon = horizon_days if horizon_days is not None else get_ml_horizon_days()
    candidates = _latest_predictions_per_ticker(
        horizon_days=horizon,
        database_url=database_url,
    )
    if not candidates:
        raise ValueError(
            f"No stored predictions for horizon {horizon}d. "
            "Run tradingai-predict-batch first."
        )

    selected = select_portfolio_positions(
        candidates,
        size=positions,
        min_combined_score=min_combined_score,
        min_probability_up=min_probability_up,
    )
    if not selected:
        raise ValueError(
            "No candidates passed portfolio filters. "
            f"Try lowering min_probability_up (current {min_probability_up})."
        )

    invested_fraction = max(0.0, min(1.0, 1.0 - cash_weight))
    holdings = _score_weights(selected, invested_fraction=invested_fraction)

    if notional_usd is not None:
        for holding in holdings:
            holding["target_usd"] = round(notional_usd * holding["weight"], 2)

    latest_ts = max(row["timestamp"] for row in selected if row.get("timestamp"))
    regime = selected[0].get("regime") if selected else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "predictions:latest_per_ticker",
        "horizon_days": horizon,
        "regime": regime,
        "universe_size": len(candidates),
        "filters": {
            "min_combined_score": min_combined_score,
            "min_probability_up": min_probability_up,
            "max_per_sector": DEFAULT_MAX_PER_SECTOR,
            "max_broad_market_etf": DEFAULT_MAX_BROAD_ETF,
            "positions": positions,
            "cash_weight": cash_weight,
        },
        "notional_usd": notional_usd,
        "cash_weight": cash_weight,
        "cash_weight_pct": round(cash_weight * 100, 2),
        "prediction_snapshot": latest_ts.isoformat() if latest_ts else None,
        "holdings": holdings,
        "disclaimer": (
            "Paper-trading starter basket from stored fusion/ML predictions. "
            "Not financial advice. Outcomes are not yet evaluated for fresh predictions."
        ),
    }
