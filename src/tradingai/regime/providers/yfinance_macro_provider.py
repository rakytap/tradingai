from __future__ import annotations

import time
from typing import Any

import pandas as pd
import yfinance as yf

from tradingai.regime.analysis.historical import macro_features_as_of
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

        as_of = spy.index[-1].date()
        return macro_features_as_of(
            spy=spy,
            vix=vix,
            hyg=hyg,
            lqd=lqd,
            as_of=as_of,
            tnx=tnx,
            sources_used=[self.name],
        )
