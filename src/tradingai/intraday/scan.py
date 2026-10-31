from __future__ import annotations

from datetime import date, datetime

from tradingai.intraday.bars import load_or_fetch_session_bars, prior_daily_close
from tradingai.intraday.bounce import assess_bounce
from tradingai.intraday.config import (
    ET,
    get_buy_notional_usd,
    get_dip_pct,
    get_dip_window_minutes,
    get_profit_taker_buffer,
    get_sell_trail,
    get_target_return,
)
from tradingai.intraday.open_dip import detect_open_dip
from tradingai.intraday.profit_target import target_sell_price
from tradingai.intraday.watchlist import bullish_sector_watchlist


DISCLAIMER = (
    "Intraday open-dip bounce scan from 1m bars (DB via tradingai-sync-intraday / IBKR TWS; --live uses yfinance). "
    "profit_taker_price nets target_return + 0.02% after IBKR Tiered buy+sell fees. "
    "Trail activates at fee-aware target_return/2 with trail distance target_return/2. "
    "Informational only — not financial advice."
)


def _attach_dip_model_prob(row: dict, *, bars, dip) -> None:
    """Add P(recover / profit target) from the Alpaca-trained dip checkpoint."""
    try:
        from tradingai.intraday.ml.infer import get_dip_predictor

        prior = dip.prior_close
        if prior is None or float(prior) <= 0:
            return
        pred = get_dip_predictor().predict_session(
            bars,
            prior_close=float(prior),
            trough_time=getattr(dip, "trough_time", None),
        )
        if pred is None:
            return
        row["recover_prob"] = pred.recover_prob
        row["dip_model_probs"] = pred.probs
        row["dip_model_side"] = pred.side
        row["dip_model_checkpoint"] = pred.checkpoint
    except Exception as exc:
        row["dip_model_error"] = str(exc)


def _row_payload(
    dip,
    bounce,
    *,
    buy_notional_usd: float,
    target_return: float,
    sell_trail: float,
    profit_taker_buffer: float,
    bars=None,
) -> dict:
    row = {
        "ticker": dip.ticker,
        "dip_pct": dip.dip_pct,
        "trough_price": dip.trough_price,
        "prior_close": dip.prior_close,
        "session_open": dip.session_open,
        "status": bounce.status,
        "bounce_score": bounce.bounce_score,
        "last_price": bounce.last_price,
        "vwap": bounce.vwap,
        "reclaim_vwap": bounce.reclaim_vwap,
        "reclaim_open": bounce.reclaim_open,
        "first_bar_green": bounce.first_bar_green,
        "trough_hammer": bounce.trough_hammer,
        "bullish_engulfing": bounce.bullish_engulfing,
        "bearish_engulfing": bounce.bearish_engulfing,
        "morning_star": bounce.morning_star,
        "momentum_15m_pct": bounce.momentum_15m_pct,
        "reasons": list(dict.fromkeys([*dip.reasons, *bounce.reasons])),
    }
    if bounce.last_price is not None and bounce.last_price > 0:
        row["profit_target"] = target_sell_price(
            bounce.last_price,
            buy_notional_usd=buy_notional_usd,
            target_return=target_return,
            profit_taker_buffer=profit_taker_buffer,
            sell_trail=sell_trail,
        )
    if bars is not None:
        _attach_dip_model_prob(row, bars=bars, dip=dip)
    return row


def evaluate_ticker(
    ticker: str,
    *,
    dip_pct: float | None = None,
    window_minutes: int | None = None,
    prefer_db: bool = True,
    buy_notional_usd: float | None = None,
    target_return: float | None = None,
    sell_trail: float | None = None,
    profit_taker_buffer: float | None = None,
    session_day: date | None = None,
) -> dict | None:
    threshold = float(dip_pct) if dip_pct is not None else get_dip_pct()
    window = int(window_minutes) if window_minutes is not None else get_dip_window_minutes()
    notional = (
        float(buy_notional_usd)
        if buy_notional_usd is not None
        else get_buy_notional_usd()
    )
    ret = float(target_return) if target_return is not None else get_target_return()
    if sell_trail is not None:
        trail = float(sell_trail)
    elif target_return is not None:
        trail = ret / 2.0
    else:
        trail = get_sell_trail()
    buffer = (
        float(profit_taker_buffer)
        if profit_taker_buffer is not None
        else get_profit_taker_buffer()
    )
    day = session_day or datetime.now(ET).date()
    bars = load_or_fetch_session_bars(ticker, session_day=day, prefer_db=prefer_db)
    if bars.empty:
        return None
    prior = prior_daily_close(ticker, before=day)
    dip = detect_open_dip(
        bars,
        ticker=ticker,
        prior_close=prior,
        dip_pct=threshold,
        window_minutes=window,
    )
    if not dip.qualifies:
        return None
    bounce = assess_bounce(bars, dip)
    return _row_payload(
        dip,
        bounce,
        buy_notional_usd=notional,
        target_return=ret,
        sell_trail=trail,
        profit_taker_buffer=buffer,
        bars=bars,
    )


def scan_open_dip_bounces(
    *,
    tickers: list[str] | None = None,
    dip_pct: float | None = None,
    window_minutes: int | None = None,
    prefer_db: bool = True,
    cap: int | None = None,
    buy_notional_usd: float | None = None,
    target_return: float | None = None,
    sell_trail: float | None = None,
    profit_taker_buffer: float | None = None,
    session_day: date | str | None = None,
) -> dict:
    """Scan watchlist for morning open dips and rank bounce candidates."""
    universe = bullish_sector_watchlist(tickers=tickers, cap=cap)
    notional = (
        float(buy_notional_usd)
        if buy_notional_usd is not None
        else get_buy_notional_usd()
    )
    ret = float(target_return) if target_return is not None else get_target_return()
    # Default trail distance = target_return / 2.
    trail = float(sell_trail) if sell_trail is not None else (
        get_sell_trail() if target_return is None else ret / 2.0
    )
    buffer = (
        float(profit_taker_buffer)
        if profit_taker_buffer is not None
        else get_profit_taker_buffer()
    )
    if isinstance(session_day, str):
        day: date | None = date.fromisoformat(session_day)
    else:
        day = session_day
    candidates: list[dict] = []
    plunging: list[dict] = []
    errors: list[str] = []

    for ticker in universe:
        try:
            payload = evaluate_ticker(
                ticker,
                dip_pct=dip_pct,
                window_minutes=window_minutes,
                prefer_db=prefer_db,
                buy_notional_usd=notional,
                target_return=ret,
                sell_trail=trail,
                profit_taker_buffer=buffer,
                session_day=day,
            )
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            continue
        if payload is None:
            continue
        if payload["status"] == "plunging":
            plunging.append(payload)
        else:
            candidates.append(payload)

    # Deepest dip first among bounce candidates; then higher bounce score.
    candidates.sort(key=lambda row: (row["dip_pct"], -row["bounce_score"]))
    plunging.sort(key=lambda row: row["dip_pct"])

    return {
        "as_of": datetime.now(ET).isoformat(),
        "session_day": (day or datetime.now(ET).date()).isoformat(),
        "universe_size": len(universe),
        "dip_pct": float(dip_pct) if dip_pct is not None else get_dip_pct(),
        "window_minutes": int(window_minutes)
        if window_minutes is not None
        else get_dip_window_minutes(),
        "buy_notional_usd": notional,
        "target_return": ret,
        "target_return_pct": round(ret * 100.0, 6),
        "profit_taker_buffer": buffer,
        "profit_taker_buffer_pct": round(buffer * 100.0, 6),
        "trail_activate_return": ret / 2.0,
        "trail_activate_return_pct": round(ret / 2.0 * 100.0, 6),
        "sell_trail": trail,
        "sell_trail_pct": round(trail * 100.0, 6),
        "candidates": candidates,
        "plunging": plunging,
        "errors": errors[:25],
        "disclaimer": DISCLAIMER,
    }
