from __future__ import annotations

from datetime import date

import pandas as pd

from tradingai.regime.analysis.scores import REGIME_SCORES
from tradingai.regime.analysis.classifier import _trend_label, _vol_label, classify_regime
from tradingai.regime.models import MacroFeatures

VIX_LOOKBACK_DAYS = 252
SPY_SMA_50_DAYS = 50
SPY_SMA_200_DAYS = 200
SPY_RETURN_6M_DAYS = 126


def percentile_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).mean() * 100.0)


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(index))
    if parsed.tz is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _series_as_of(series: pd.Series, as_of: date) -> pd.Series:
    if series.empty:
        return series
    normalized = series.copy()
    normalized.index = _normalize_index(normalized.index)
    cutoff = pd.Timestamp(as_of).normalize()
    return normalized.loc[normalized.index <= cutoff]


def _latest_value(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def macro_features_as_of(
    *,
    spy: pd.Series,
    vix: pd.Series,
    hyg: pd.Series,
    lqd: pd.Series,
    as_of: date,
    tnx: pd.Series | None = None,
    sources_used: list[str] | None = None,
) -> MacroFeatures:
    spy_slice = _series_as_of(spy, as_of)
    vix_slice = _series_as_of(vix, as_of)
    hyg_slice = _series_as_of(hyg, as_of)
    lqd_slice = _series_as_of(lqd, as_of)
    tnx_slice = _series_as_of(tnx, as_of) if tnx is not None else pd.Series(dtype=float)

    spy_close = _latest_value(spy_slice)
    spy_sma_50 = float(spy_slice.tail(SPY_SMA_50_DAYS).mean()) if len(spy_slice) >= SPY_SMA_50_DAYS else None
    spy_sma_200 = (
        float(spy_slice.tail(SPY_SMA_200_DAYS).mean())
        if len(spy_slice) >= SPY_SMA_200_DAYS
        else None
    )

    spy_return_6m = None
    if len(spy_slice) > 1:
        lookback = min(SPY_RETURN_6M_DAYS, len(spy_slice) - 1)
        if lookback > 0:
            spy_return_6m = float(
                (spy_slice.iloc[-1] / spy_slice.iloc[-lookback] - 1.0) * 100.0
            )

    vix_value = _latest_value(vix_slice)
    vix_pct = None
    if vix_value is not None:
        vix_window = vix_slice.tail(VIX_LOOKBACK_DAYS)
        vix_pct = percentile_rank(vix_window, vix_value)

    hyg_lqd_ratio = None
    hyg_value = _latest_value(hyg_slice)
    lqd_value = _latest_value(lqd_slice)
    if hyg_value is not None and lqd_value is not None and lqd_value != 0:
        hyg_lqd_ratio = hyg_value / lqd_value

    treasury_10y = _latest_value(tnx_slice) if not tnx_slice.empty else None

    return MacroFeatures(
        vix=vix_value,
        vix_percentile_1y=vix_pct,
        spy_close=spy_close,
        spy_sma_50=spy_sma_50,
        spy_sma_200=spy_sma_200,
        spy_return_6m_pct=spy_return_6m,
        treasury_10y=treasury_10y,
        hyg_lqd_ratio=hyg_lqd_ratio,
        sources_used=sources_used or [],
    )


def macro_features_to_ml_row(features: MacroFeatures) -> dict[str, float | None]:
    regime_label = classify_regime(features)
    trend = _trend_label(features)
    vol = _vol_label(features)

    spy_close_to_sma200 = None
    if features.spy_close is not None and features.spy_sma_200 not in (None, 0):
        spy_close_to_sma200 = features.spy_close / features.spy_sma_200

    return {
        "regime_vix": features.vix,
        "regime_vix_pct_1y": features.vix_percentile_1y,
        "regime_spy_return_6m_pct": features.spy_return_6m_pct,
        "regime_spy_close_to_sma200": spy_close_to_sma200,
        "regime_hyg_lqd_ratio": features.hyg_lqd_ratio,
        "regime_score": REGIME_SCORES.get(regime_label, 0.0),
        "regime_high_vol": 1.0 if vol == "high_volatility" else 0.0,
        "regime_bull_trend": 1.0 if trend == "bull" else 0.0,
    }


def build_regime_feature_frame(
    index: pd.DatetimeIndex,
    macro_histories: dict[str, pd.Series],
) -> pd.DataFrame:
    spy = macro_histories.get("SPY", pd.Series(dtype=float))
    vix = macro_histories.get("VIX", pd.Series(dtype=float))
    hyg = macro_histories.get("HYG", pd.Series(dtype=float))
    lqd = macro_histories.get("LQD", pd.Series(dtype=float))
    tnx = macro_histories.get("TNX")

    rows: list[dict] = []
    for timestamp in index:
        as_of = timestamp.date() if hasattr(timestamp, "date") else pd.Timestamp(timestamp).date()
        features = macro_features_as_of(
            spy=spy,
            vix=vix,
            hyg=hyg,
            lqd=lqd,
            as_of=as_of,
            tnx=tnx,
            sources_used=["historical"],
        )
        row = macro_features_to_ml_row(features)
        row["date"] = as_of
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
