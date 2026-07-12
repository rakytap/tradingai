from __future__ import annotations

import time
from typing import Any

import pandas as pd

from tradingai.regime.config import get_fred_api_key
from tradingai.regime.models import MacroFeatures
from tradingai.regime.providers.base import MacroDataProvider

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 900

FRED_SERIES = {
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "fed_funds_rate": "FEDFUNDS",
    "inflation_breakeven_10y": "T10YIE",
    "cpi": "CPIAUCSL",
    "credit_spread_hy_oas": "BAMLH0A0HYM2",
}


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


def _percentile_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).mean() * 100.0)


class FredMacroProvider(MacroDataProvider):
    name = "fred"

    def _client(self):
        api_key = get_fred_api_key()
        if not api_key:
            raise ValueError(
                "FRED_API_KEY is not configured. Add it to .env to enable FRED macro data."
            )
        from fredapi import Fred

        return Fred(api_key=api_key)

    def _series(self, series_id: str, observation_start: str = "2020-01-01") -> pd.Series:
        cache_key = f"fred:{series_id}:{observation_start}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached.copy()
        client = self._client()
        data = client.get_series(series_id, observation_start=observation_start)
        _cache_set(cache_key, data)
        return data.copy()

    def fetch_features(self) -> MacroFeatures:
        treasury_10y_series = self._series(FRED_SERIES["treasury_10y"])
        treasury_2y_series = self._series(FRED_SERIES["treasury_2y"])
        fed_funds_series = self._series(FRED_SERIES["fed_funds_rate"])
        breakeven_series = self._series(FRED_SERIES["inflation_breakeven_10y"])
        cpi_series = self._series(FRED_SERIES["cpi"], observation_start="2015-01-01")
        credit_series = self._series(FRED_SERIES["credit_spread_hy_oas"])

        treasury_10y = float(treasury_10y_series.dropna().iloc[-1])
        treasury_2y = float(treasury_2y_series.dropna().iloc[-1])
        spread_bps = (treasury_10y - treasury_2y) * 100.0

        cpi_clean = cpi_series.dropna()
        cpi_yoy = None
        if len(cpi_clean) >= 13:
            cpi_yoy = float((cpi_clean.iloc[-1] / cpi_clean.iloc[-13] - 1.0) * 100.0)

        credit_value = float(credit_series.dropna().iloc[-1])
        credit_pct = _percentile_rank(credit_series.tail(252), credit_value)

        return MacroFeatures(
            treasury_10y=treasury_10y,
            treasury_2y=treasury_2y,
            yield_spread_2s10s_bps=spread_bps,
            yield_curve_inverted=spread_bps < 0,
            fed_funds_rate=float(fed_funds_series.dropna().iloc[-1]),
            inflation_breakeven_10y=float(breakeven_series.dropna().iloc[-1]),
            cpi_yoy_pct=cpi_yoy,
            credit_spread_hy_oas=credit_value,
            credit_spread_percentile_1y=credit_pct,
            sources_used=[self.name],
        )
