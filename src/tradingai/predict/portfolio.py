from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tradingai.analysis.ml.config import is_l5_ranking_enabled
from tradingai.analysis.ml.ohlcv import load_ticker_ohlcv
from tradingai.data.connection import get_connection
from tradingai.data.etf_symbols import is_etf
from tradingai.predict.config import get_ml_horizon_days, get_portfolio_rank_key
from tradingai.predict.ibkr_fees import (
    estimate_rebalance_commission_from_weights,
    tier_rate_per_share,
)
from tradingai.predict.portfolio_risk import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_AVERSION,
    annualized_volatility_pct,
    build_returns_frame,
    estimate_mu_sigma,
    optimize_markowitz_rebalance_weights,
    optimize_markowitz_weights,
)

BROAD_MARKET_ETFS = frozenset({"SPY", "VOO", "IVV", "VTI", "QQQ", "DIA", "IWM", "RSP"})
DEFAULT_POSITIONS = 10
DEFAULT_CASH_WEIGHT = 0.10
DEFAULT_MIN_COMBINED_SCORE = 0.0
DEFAULT_MIN_PROBABILITY_UP = 0.55
DEFAULT_MAX_PER_SECTOR = 10
DEFAULT_MAX_BROAD_ETF = 1
DEFAULT_OPTIMIZER = "score"
DEFAULT_PREVIOUS_PORTFOLIO_PATH = Path("data/paper_portfolio.json")
DEFAULT_COMMISSION_PREFERENCE = 1.0
IBKR_REBALANCE_OPTIMIZER = "markowitz_ibkr_rebalance"


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
                   model_name, actual_return_pct, direction_correct,
                   ml_rank_score, sector_rank
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


def _rank_key(row: dict) -> float:
    key = get_portfolio_rank_key()
    if key == "ml_rank_score":
        ml_rank = row.get("ml_rank_score")
        if ml_rank is not None:
            try:
                return float(ml_rank)
            except (TypeError, ValueError):
                pass
        return float(row["combined_score"])
    return float(row["combined_score"])


def _load_bullish_sectors() -> set[str]:
    """Canonical L4 board sectors currently tagged bullish."""
    from tradingai.analysis.ml.ranking import bullish_sectors_from_regime
    from tradingai.regime.sector_proxies import normalize_sector_label

    labels: set[str] = set()
    for sector in bullish_sectors_from_regime():
        canonical = normalize_sector_label(sector) or str(sector).strip()
        if canonical:
            labels.add(canonical)
    return labels


def resolve_candidate_sector(row: dict) -> str | None:
    """Best-effort GICS sector for a prediction row (row → stocks table)."""
    from tradingai.data.repositories.stocks import get_stock
    from tradingai.regime.sector_proxies import normalize_sector_label

    sector = normalize_sector_label(_sector_key(row))
    if sector:
        return sector
    ticker = str(row.get("ticker") or "").strip().upper()
    if not ticker:
        return None
    try:
        stock = get_stock(ticker)
    except Exception:
        stock = None
    return normalize_sector_label(stock.get("sector") if stock else None)


def passes_bullish_sector_gate(
    row: dict,
    *,
    bullish_sectors: set[str] | None = None,
) -> bool:
    """True when the ticker's sector is on the live L4 bullish board.

    Does **not** treat stored ``regime == market:bull`` as sector-bullish — that
    label is a market fallback when sector metadata was missing at predict time.
    """
    allowed = bullish_sectors if bullish_sectors is not None else _load_bullish_sectors()
    if not allowed:
        return False
    sector = resolve_candidate_sector(row)
    return bool(sector and sector in allowed)


def select_portfolio_positions(
    candidates: list[dict],
    *,
    size: int = DEFAULT_POSITIONS,
    min_combined_score: float = DEFAULT_MIN_COMBINED_SCORE,
    min_probability_up: float = DEFAULT_MIN_PROBABILITY_UP,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    max_broad_etf: int = DEFAULT_MAX_BROAD_ETF,
    require_bullish_sector: bool = False,
    bullish_sectors: set[str] | None = None,
) -> list[dict]:
    allowed_sectors = (
        bullish_sectors
        if bullish_sectors is not None
        else (_load_bullish_sectors() if require_bullish_sector else None)
    )
    filtered = []
    for row in candidates:
        if float(row["combined_score"]) <= min_combined_score:
            continue
        if float(row["probability_up"]) < min_probability_up:
            continue
        if require_bullish_sector and not passes_bullish_sector_gate(
            row,
            bullish_sectors=allowed_sectors,
        ):
            continue
        filtered.append(row)

    ranked = sorted(filtered, key=lambda row: _rank_key(row), reverse=True)

    stocks = [row for row in ranked if not is_etf(str(row["ticker"]))]
    etfs = [row for row in ranked if is_etf(str(row["ticker"]))]

    selected: list[dict] = []
    sector_counts: dict[str, int] = defaultdict(int)
    broad_etf_count = 0

    def _can_add(row: dict) -> bool:
        ticker = str(row["ticker"]).upper()
        sector = resolve_candidate_sector(row)
        if sector is not None and sector_counts[sector] >= max_per_sector:
            return False
        if ticker in BROAD_MARKET_ETFS and broad_etf_count >= max_broad_etf:
            return False
        return True

    def _add(row: dict) -> None:
        nonlocal broad_etf_count
        ticker = str(row["ticker"]).upper()
        sector = resolve_candidate_sector(row)
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


def _holding_base(row: dict, *, weight: float, optimizer: str) -> dict:
    return {
        "ticker": str(row["ticker"]).upper(),
        "weight": round(weight, 4),
        "weight_pct": round(weight * 100, 2),
        "combined_score": round(float(row["combined_score"]), 4),
        "ml_rank_score": (
            round(float(row["ml_rank_score"]), 4)
            if row.get("ml_rank_score") is not None
            else None
        ),
        "sector_rank": row.get("sector_rank"),
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
        "optimizer": optimizer,
    }


def _score_weights(rows: list[dict], *, invested_fraction: float) -> list[dict]:
    if not rows:
        return []

    scores = [max(_rank_key(row), 0.01) for row in rows]
    total = sum(scores)
    return [
        _holding_base(row, weight=invested_fraction * (score / total), optimizer="score")
        for row, score in zip(rows, scores)
    ]


def _markowitz_weights(
    rows: list[dict],
    *,
    horizon_days: int,
    invested_fraction: float,
    lookback_days: int,
    risk_aversion: float,
    max_weight: float,
    database_url: str | None = None,
) -> tuple[list[dict], str]:
    tickers = [str(row["ticker"]).upper() for row in rows]
    returns = build_returns_frame(
        tickers,
        lookback_days=lookback_days,
        database_url=database_url,
    )
    by_ticker = {str(row["ticker"]).upper(): row for row in rows}
    ordered_tickers = [ticker for ticker in returns.columns if ticker in by_ticker]
    if len(ordered_tickers) < 2:
        raise ValueError(
            "Markowitz needs at least 2 tickers with overlapping price history. "
            "Run tradingai-sync-prices."
        )
    ordered_rows = [by_ticker[ticker] for ticker in ordered_tickers]
    returns = returns[ordered_tickers]
    mu, sigma, _ = estimate_mu_sigma(
        returns,
        ordered_rows,
        horizon_days=horizon_days,
    )

    try:
        weights = optimize_markowitz_weights(
            mu,
            sigma,
            invested_fraction=invested_fraction,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
        )
        status = "ok"
    except (RuntimeError, ValueError):
        fallback = _score_weights(ordered_rows, invested_fraction=invested_fraction)
        return fallback, "fallback_score"

    holdings = []
    for index, (row, weight) in enumerate(zip(ordered_rows, weights)):
        holding = _holding_base(row, weight=float(weight), optimizer="markowitz")
        holding["mu_daily"] = round(float(mu[index]), 6)
        holding["annualized_volatility_pct"] = round(
            annualized_volatility_pct(sigma, index),
            4,
        )
        holdings.append(holding)
    return holdings, status


def build_paper_portfolio(
    *,
    horizon_days: int | None = None,
    positions: int = DEFAULT_POSITIONS,
    cash_weight: float = DEFAULT_CASH_WEIGHT,
    min_combined_score: float = DEFAULT_MIN_COMBINED_SCORE,
    min_probability_up: float = DEFAULT_MIN_PROBABILITY_UP,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    notional_usd: float | None = 100_000.0,
    optimizer: str = DEFAULT_OPTIMIZER,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    database_url: str | None = None,
    require_bullish_sector: bool | None = None,
) -> dict:
    horizon = horizon_days if horizon_days is not None else get_ml_horizon_days()
    selected_optimizer = (optimizer or DEFAULT_OPTIMIZER).lower().strip()
    if selected_optimizer not in {"score", "markowitz"}:
        raise ValueError("optimizer must be 'score' or 'markowitz'.")

    bullish_gate = (
        is_l5_ranking_enabled() if require_bullish_sector is None else require_bullish_sector
    )
    sector_cap = max(1, int(max_per_sector))
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
        max_per_sector=sector_cap,
        require_bullish_sector=bullish_gate,
    )
    if not selected:
        raise ValueError(
            "No candidates passed portfolio filters. "
            f"Try lowering min_probability_up (current {min_probability_up})"
            + (" or disabling L5 bullish-sector gate." if bullish_gate else ".")
        )

    invested_fraction = max(0.0, min(1.0, 1.0 - cash_weight))
    optimizer_status = "ok"
    if selected_optimizer == "markowitz":
        holdings, optimizer_status = _markowitz_weights(
            selected,
            horizon_days=horizon,
            invested_fraction=invested_fraction,
            lookback_days=lookback_days,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            database_url=database_url,
        )
    else:
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
        "optimizer": selected_optimizer,
        "optimizer_status": optimizer_status,
        "lookback_days": lookback_days if selected_optimizer == "markowitz" else None,
        "risk_aversion": risk_aversion if selected_optimizer == "markowitz" else None,
        "max_weight": max_weight if selected_optimizer == "markowitz" else None,
        "filters": {
            "min_combined_score": min_combined_score,
            "min_probability_up": min_probability_up,
            "max_per_sector": sector_cap,
            "max_broad_market_etf": DEFAULT_MAX_BROAD_ETF,
            "positions": positions,
            "cash_weight": cash_weight,
            "require_bullish_sector": bullish_gate,
            "rank_key": get_portfolio_rank_key(),
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


def load_previous_portfolio(path: str | Path) -> dict:
    """Load a prior paper-portfolio JSON; raises if missing or invalid."""
    portfolio_path = Path(path)
    if not portfolio_path.is_file():
        raise FileNotFoundError(
            f"Previous portfolio not found: {portfolio_path.resolve()}. "
            "Run tradingai-paper-portfolio first or pass --previous-portfolio."
        )
    payload = json.loads(portfolio_path.read_text())
    holdings = payload.get("holdings")
    if not isinstance(holdings, list) or not holdings:
        raise ValueError(f"Previous portfolio has no holdings: {portfolio_path}")
    return payload


def _latest_close(ticker: str, *, database_url: str | None = None) -> float | None:
    ohlcv = load_ticker_ohlcv(ticker, database_url=database_url)
    if ohlcv.empty or "close" not in ohlcv.columns:
        return None
    close = float(ohlcv["close"].iloc[-1])
    return close if close > 0 else None


def _row_from_prior_holding(holding: dict, *, horizon_days: int) -> dict:
    """Synthetic prediction row so prior-only names can stay in the QP."""
    return {
        "id": holding.get("prediction_id"),
        "ticker": str(holding["ticker"]).upper(),
        "timestamp": None,
        "horizon_days": horizon_days,
        "probability_up": float(holding.get("probability_up") or 0.5),
        "expected_return_pct": float(holding.get("expected_return_pct") or 0.0),
        "confidence": float(holding.get("confidence") or 0.0),
        "combined_score": float(holding.get("combined_score") or 0.0),
        "ml_rank_score": holding.get("ml_rank_score"),
        "sector_rank": holding.get("sector_rank"),
        "sector": holding.get("sector"),
        "regime": holding.get("regime"),
        "prediction_type": holding.get("prediction_type"),
        "technical_score": float(holding.get("technical_score") or 0.0),
        "news_score": float(holding.get("news_score") or 0.0),
        "regime_score": float(holding.get("regime_score") or 0.0),
        "model_name": holding.get("model_name"),
    }


def build_paper_portfolio_ibkr_rebalance(
    *,
    previous_portfolio_path: str | Path = DEFAULT_PREVIOUS_PORTFOLIO_PATH,
    commission_preference: float = DEFAULT_COMMISSION_PREFERENCE,
    monthly_volume_shares: float = 0.0,
    horizon_days: int | None = None,
    positions: int = DEFAULT_POSITIONS,
    cash_weight: float = DEFAULT_CASH_WEIGHT,
    min_combined_score: float = DEFAULT_MIN_COMBINED_SCORE,
    min_probability_up: float = DEFAULT_MIN_PROBABILITY_UP,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    notional_usd: float = 100_000.0,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    database_url: str | None = None,
    require_bullish_sector: bool | None = None,
) -> dict:
    """Reoptimize a prior paper portfolio with soft IBKR Tiered commission cost."""
    prior_path = Path(previous_portfolio_path)
    prior = load_previous_portfolio(prior_path)
    horizon = horizon_days if horizon_days is not None else get_ml_horizon_days()
    bullish_gate = (
        is_l5_ranking_enabled() if require_bullish_sector is None else require_bullish_sector
    )
    sector_cap = max(1, int(max_per_sector))
    notional = float(notional_usd)
    if notional <= 0:
        raise ValueError("notional_usd must be positive for IBKR rebalance.")

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
        max_per_sector=sector_cap,
        require_bullish_sector=bullish_gate,
    )
    if not selected:
        raise ValueError(
            "No candidates passed portfolio filters. "
            f"Try lowering min_probability_up (current {min_probability_up})"
            + (" or disabling L5 bullish-sector gate." if bullish_gate else ".")
        )

    prior_by_ticker: dict[str, dict] = {}
    prior_weights: dict[str, float] = {}
    for holding in prior["holdings"]:
        ticker = str(holding["ticker"]).upper()
        prior_by_ticker[ticker] = holding
        prior_weights[ticker] = float(holding.get("weight") or 0.0)

    by_ticker: dict[str, dict] = {
        str(row["ticker"]).upper(): row for row in selected
    }
    for ticker, holding in prior_by_ticker.items():
        if ticker not in by_ticker:
            by_ticker[ticker] = _row_from_prior_holding(holding, horizon_days=horizon)

    tickers = sorted(by_ticker)
    returns = build_returns_frame(
        tickers,
        lookback_days=lookback_days,
        database_url=database_url,
    )
    ordered_tickers = [ticker for ticker in returns.columns if ticker in by_ticker]
    dropped = [ticker for ticker in tickers if ticker not in ordered_tickers]

    prices: dict[str, float] = {}
    usable: list[str] = []
    for ticker in ordered_tickers:
        close = _latest_close(ticker, database_url=database_url)
        if close is None:
            dropped.append(ticker)
            continue
        prices[ticker] = close
        usable.append(ticker)
    if len(usable) < 2:
        raise ValueError(
            "IBKR rebalance needs at least 2 tickers with overlapping returns and prices. "
            f"Dropped: {sorted(set(dropped))}. Run tradingai-sync-prices."
        )

    ordered_rows = [by_ticker[ticker] for ticker in usable]
    returns = returns[usable]
    mu, sigma, _ = estimate_mu_sigma(
        returns,
        ordered_rows,
        horizon_days=horizon,
    )
    w0 = np.array(
        [prior_weights.get(ticker, 0.0) for ticker in usable],
        dtype=float,
    )
    # Renormalize prior onto the invested sleeve if it only covered a subset.
    invested_fraction = max(0.0, min(1.0, 1.0 - cash_weight))
    prior_sum = float(w0.sum())
    if prior_sum > 1e-12:
        w0 = w0 * (invested_fraction / prior_sum)
    price_vec = np.array(
        [prices[ticker] for ticker in usable],
        dtype=float,
    )
    rate = tier_rate_per_share(monthly_volume_shares)

    try:
        weights = optimize_markowitz_rebalance_weights(
            mu,
            sigma,
            w0,
            price_vec,
            invested_fraction=invested_fraction,
            notional_usd=notional,
            commission_preference=float(commission_preference),
            rate_per_share=rate,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
        )
        optimizer_status = "ok"
    except (RuntimeError, ValueError):
        weights = optimize_markowitz_weights(
            mu,
            sigma,
            invested_fraction=invested_fraction,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
        )
        optimizer_status = "fallback_markowitz"

    new_weights = {
        ticker: float(weight) for ticker, weight in zip(usable, weights)
    }
    total_fee, trade_details = estimate_rebalance_commission_from_weights(
        tickers=usable,
        prior_weights={t: float(w) for t, w in zip(usable, w0)},
        new_weights=new_weights,
        prices=prices,
        notional_usd=notional,
        monthly_volume_shares=monthly_volume_shares,
    )
    detail_by_ticker = {item["ticker"]: item for item in trade_details}

    holdings = []
    for index, (row, weight) in enumerate(zip(ordered_rows, weights)):
        ticker = usable[index]
        holding = _holding_base(
            row,
            weight=float(weight),
            optimizer=IBKR_REBALANCE_OPTIMIZER,
        )
        holding["mu_daily"] = round(float(mu[index]), 6)
        holding["annualized_volatility_pct"] = round(
            annualized_volatility_pct(sigma, index),
            4,
        )
        holding["target_usd"] = round(notional * float(weight), 2)
        detail = detail_by_ticker.get(ticker, {})
        holding["prior_weight"] = round(float(detail.get("prior_weight", 0.0)), 4)
        holding["trade_usd"] = detail.get("trade_usd", 0.0)
        holding["est_commission_usd"] = detail.get("est_commission_usd", 0.0)
        holdings.append(holding)

    latest_ts = max(
        (row["timestamp"] for row in selected if row.get("timestamp")),
        default=None,
    )
    regime = selected[0].get("regime") if selected else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "predictions:latest_per_ticker+prior_portfolio",
        "horizon_days": horizon,
        "regime": regime,
        "universe_size": len(candidates),
        "optimizer": IBKR_REBALANCE_OPTIMIZER,
        "optimizer_status": optimizer_status,
        "lookback_days": lookback_days,
        "risk_aversion": risk_aversion,
        "max_weight": max_weight,
        "commission_preference": float(commission_preference),
        "monthly_volume_shares": float(monthly_volume_shares),
        "ibkr_tier_rate_per_share": rate,
        "estimated_ibkr_commission_usd": round(total_fee, 4),
        "previous_portfolio_path": str(prior_path),
        "dropped_tickers": sorted(set(dropped)),
        "filters": {
            "min_combined_score": min_combined_score,
            "min_probability_up": min_probability_up,
            "max_per_sector": sector_cap,
            "max_broad_market_etf": DEFAULT_MAX_BROAD_ETF,
            "positions": positions,
            "cash_weight": cash_weight,
            "require_bullish_sector": bullish_gate,
            "rank_key": get_portfolio_rank_key(),
        },
        "notional_usd": notional,
        "cash_weight": cash_weight,
        "cash_weight_pct": round(cash_weight * 100, 2),
        "prediction_snapshot": latest_ts.isoformat() if latest_ts else None,
        "holdings": holdings,
        "disclaimer": (
            "Paper-trading IBKR Tiered soft-cost rebalance from stored predictions "
            "and a prior paper portfolio. Broker commission estimate only "
            "(no exchange/regulatory fees). Not financial advice."
        ),
    }
