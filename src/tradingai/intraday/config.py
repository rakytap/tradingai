from __future__ import annotations

import os
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RTH_OPEN_HOUR = 9
RTH_OPEN_MINUTE = 30
DEFAULT_DIP_WINDOW_MINUTES = 30
DEFAULT_DIP_PCT = 2.0
DEFAULT_WATCHLIST_CAP = 1000
DEFAULT_BAR_INTERVAL = "1m"
DEFAULT_BAR_PERIOD = "5d"
DEFAULT_KEEP_DAYS = 5
DEFAULT_IBKR_HOST = "127.0.0.1"
DEFAULT_IBKR_PORT = 7497  # TWS paper
DEFAULT_IBKR_CLIENT_ID = 7
DEFAULT_IBKR_PACE_SECONDS = 0.35
# Live order placement via tradingai-ibkr MCP (off by default).
DEFAULT_IBKR_ALLOW_ORDERS = False
DEFAULT_IBKR_MAX_ORDER_SHARES = 1_000.0
DEFAULT_MOMENTUM_MINUTES = 15
DEFAULT_LOWER_LOW_BARS = 5
# Penalize bounce score when the opening N-minute composite bar is green (close > open).
DEFAULT_FIRST_BAR_MINUTES = 15
DEFAULT_FIRST_BAR_GREEN_PENALTY = 0.25
# Japanese candle confirmation (5m patterns; ranking only).
DEFAULT_CANDLE_HAMMER_LOWER_WICK_RATIO = 2.0
DEFAULT_CANDLE_HAMMER_MAX_UPPER_WICK_RATIO = 0.5
DEFAULT_CANDLE_MIN_REL_VOLUME = 0.5
DEFAULT_CANDLE_SCORE_HAMMER = 0.30
DEFAULT_CANDLE_SCORE_BULLISH_ENGULF = 0.25
DEFAULT_CANDLE_SCORE_BEARISH_ENGULF = 0.30
DEFAULT_BUY_NOTIONAL_USD = 20_000.0
# (1/365)*(7/5) ≈ 0.3836% net target return on buy notional
DEFAULT_TARGET_RETURN = (1.0 / 365.0) * (7.0 / 5.0)
# Sell trail order % (fraction). None → target_return / 2.
DEFAULT_SELL_TRAIL: float | None = None
# Extra cushion on profit-taker beyond target_return (fraction).
DEFAULT_PROFIT_TAKER_BUFFER = 0.0002  # 0.02%


@lru_cache
def get_buy_notional_usd() -> float:
    raw = os.getenv("TRADINGAI_OPEN_DIP_BUY_NOTIONAL")
    if not raw:
        return DEFAULT_BUY_NOTIONAL_USD
    return max(1.0, float(raw))


@lru_cache
def get_target_return() -> float:
    raw = os.getenv("TRADINGAI_OPEN_DIP_TARGET_RETURN")
    if not raw:
        return DEFAULT_TARGET_RETURN
    return max(0.0, float(raw))


@lru_cache
def get_sell_trail() -> float:
    """Trailing sell-order distance as a fraction (default = target_return / 2)."""
    raw = os.getenv("TRADINGAI_OPEN_DIP_SELL_TRAIL")
    if not raw:
        if DEFAULT_SELL_TRAIL is not None:
            return max(0.0, float(DEFAULT_SELL_TRAIL))
        return get_target_return() / 2.0
    return max(0.0, float(raw))


@lru_cache
def get_profit_taker_buffer() -> float:
    raw = os.getenv("TRADINGAI_OPEN_DIP_PROFIT_TAKER_BUFFER")
    if not raw:
        return DEFAULT_PROFIT_TAKER_BUFFER
    return max(0.0, float(raw))


@lru_cache
def get_dip_window_minutes() -> int:
    raw = os.getenv("TRADINGAI_OPEN_DIP_WINDOW_MINUTES")
    if not raw:
        return DEFAULT_DIP_WINDOW_MINUTES
    return max(5, int(raw))


@lru_cache
def get_dip_pct() -> float:
    raw = os.getenv("TRADINGAI_OPEN_DIP_PCT")
    if not raw:
        return DEFAULT_DIP_PCT
    return max(0.1, float(raw))


@lru_cache
def get_watchlist_cap() -> int:
    raw = os.getenv("TRADINGAI_OPEN_DIP_WATCHLIST_CAP")
    if not raw:
        return DEFAULT_WATCHLIST_CAP
    return max(1, int(raw))


@lru_cache
def get_bar_interval() -> str:
    return (os.getenv("TRADINGAI_INTRADAY_INTERVAL") or DEFAULT_BAR_INTERVAL).strip()


@lru_cache
def get_bar_period() -> str:
    return (os.getenv("TRADINGAI_INTRADAY_PERIOD") or DEFAULT_BAR_PERIOD).strip()


@lru_cache
def get_ibkr_host() -> str:
    return (os.getenv("TRADINGAI_IBKR_HOST") or DEFAULT_IBKR_HOST).strip()


@lru_cache
def get_ibkr_port() -> int:
    raw = os.getenv("TRADINGAI_IBKR_PORT")
    if not raw:
        return DEFAULT_IBKR_PORT
    return int(raw)


@lru_cache
def get_ibkr_client_id() -> int:
    raw = os.getenv("TRADINGAI_IBKR_CLIENT_ID")
    if not raw:
        return DEFAULT_IBKR_CLIENT_ID
    return int(raw)


@lru_cache
def get_ibkr_mcp_client_id() -> int:
    """Separate client id for tradingai-ibkr MCP (avoids clash with sync)."""
    raw = os.getenv("TRADINGAI_IBKR_MCP_CLIENT_ID")
    if not raw:
        return get_ibkr_client_id() + 1
    return int(raw)


@lru_cache
def get_ibkr_pace_seconds() -> float:
    raw = os.getenv("TRADINGAI_IBKR_PACE_SECONDS")
    if not raw:
        return DEFAULT_IBKR_PACE_SECONDS
    return max(0.0, float(raw))


@lru_cache
def get_ibkr_allow_orders() -> bool:
    """Gate for MCP buy/sell placement (requires TRADINGAI_IBKR_ALLOW_ORDERS=1)."""
    raw = (os.getenv("TRADINGAI_IBKR_ALLOW_ORDERS") or "").strip().lower()
    if not raw:
        return DEFAULT_IBKR_ALLOW_ORDERS
    return raw in {"1", "true", "yes", "on"}


@lru_cache
def get_ibkr_max_order_shares() -> float:
    raw = os.getenv("TRADINGAI_IBKR_MAX_ORDER_SHARES")
    if not raw:
        return DEFAULT_IBKR_MAX_ORDER_SHARES
    return max(0.0, float(raw))


@lru_cache
def get_candle_hammer_weight() -> float:
    raw = os.getenv("TRADINGAI_CANDLE_SCORE_HAMMER")
    if not raw:
        return DEFAULT_CANDLE_SCORE_HAMMER
    return max(0.0, float(raw))


@lru_cache
def get_candle_bullish_engulf_weight() -> float:
    raw = os.getenv("TRADINGAI_CANDLE_SCORE_BULLISH_ENGULF")
    if not raw:
        return DEFAULT_CANDLE_SCORE_BULLISH_ENGULF
    return max(0.0, float(raw))


@lru_cache
def get_candle_bearish_engulf_weight() -> float:
    raw = os.getenv("TRADINGAI_CANDLE_SCORE_BEARISH_ENGULF")
    if not raw:
        return DEFAULT_CANDLE_SCORE_BEARISH_ENGULF
    return max(0.0, float(raw))


@lru_cache
def get_candle_min_rel_volume() -> float:
    raw = os.getenv("TRADINGAI_CANDLE_MIN_REL_VOLUME")
    if not raw:
        return DEFAULT_CANDLE_MIN_REL_VOLUME
    return max(0.0, float(raw))


def clear_intraday_config_cache() -> None:
    get_dip_window_minutes.cache_clear()
    get_dip_pct.cache_clear()
    get_watchlist_cap.cache_clear()
    get_bar_interval.cache_clear()
    get_bar_period.cache_clear()
    get_buy_notional_usd.cache_clear()
    get_target_return.cache_clear()
    get_sell_trail.cache_clear()
    get_profit_taker_buffer.cache_clear()
    get_ibkr_host.cache_clear()
    get_ibkr_port.cache_clear()
    get_ibkr_client_id.cache_clear()
    get_ibkr_mcp_client_id.cache_clear()
    get_ibkr_pace_seconds.cache_clear()
    get_ibkr_allow_orders.cache_clear()
    get_ibkr_max_order_shares.cache_clear()
    get_candle_hammer_weight.cache_clear()
    get_candle_bullish_engulf_weight.cache_clear()
    get_candle_bearish_engulf_weight.cache_clear()
    get_candle_min_rel_volume.cache_clear()
