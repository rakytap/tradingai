from __future__ import annotations

import time
from typing import Any

import pandas as pd
import yfinance as yf

from tradingai.regime.models import MacroFeatures
from tradingai.regime.providers.base import MacroDataProvider

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 900


def _cache_get(key: str) -> Any | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, value)


def _normalize_close(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("No price data returned.")
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0]
        return close
    return df["Close"]


def _percentile_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).mean() * 100.0)


class YFinanceMacroProvider(MacroDataProvider):
    name = "yfinance"

    def _history(self, symbol: str, period: str = "1y") -> pd.Series:
        cache_key = f"hist:{symbol}:{period}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached.copy()
        ticker = yf.Ticker(symbol)
        raw = ticker.history(period=period, interval="1d", auto_adjust=True)
        close = _normalize_close(raw)
        _cache_set(cache_key, close)
        return close.copy()

    def fetch_features(self) -> MacroFeatures:
        spy = self._history("SPY", period="2y")
        vix = self._history("^VIX", period="1y")
        tnx = self._history("^TNX", period="1y")
        hyg = self._history("HYG", period="1y")
        lqd = self._history("LQD", period="1y")

        spy_close = float(spy.iloc[-1])
        spy_sma_50 = float(spy.tail(50).mean())
        spy_sma_200 = float(spy.tail(200).mean()) if len(spy) >= 200 else None
        lookback = min(126, len(spy) - 1)
        spy_return_6m = None
        if lookback > 0:
            spy_return_6m = float((spy.iloc[-1] / spy.iloc[-lookback] - 1.0) * 100.0)

        vix_value = float(vix.iloc[-1])
        vix_pct = _percentile_rank(vix, vix_value)

        hyg_lqd_ratio = None
        if not hyg.empty and not lqd.empty:
            hyg_lqd_ratio = float(hyg.iloc[-1] / lqd.iloc[-1])

        return MacroFeatures(
            vix=vix_value,
            vix_percentile_1y=vix_pct,
            spy_close=spy_close,
            spy_sma_50=spy_sma_50,
            spy_sma_200=spy_sma_200,
            spy_return_6m_pct=spy_return_6m,
            treasury_10y=float(tnx.iloc[-1]) if not tnx.empty else None,
            hyg_lqd_ratio=hyg_lqd_ratio,
            sources_used=[self.name],
        )
