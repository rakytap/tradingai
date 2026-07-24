from __future__ import annotations

import pandas as pd

MIN_CS_PEERS = 5


def add_direction_label(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    result = frame.copy()
    future_close = result["close"].shift(-horizon_days)
    forward_return_pct = (future_close / result["close"] - 1.0) * 100.0
    result["forward_return_pct"] = forward_return_pct
    result["direction_up"] = forward_return_pct > 0
    return result


def add_cross_section_label(
    frame: pd.DataFrame,
    *,
    sector_col: str = "sector",
    min_peers: int = MIN_CS_PEERS,
) -> pd.DataFrame:
    """Label rows relative to same-date sector peers (above median = 1)."""
    if frame.empty:
        return frame.copy()
    if "forward_return_pct" not in frame.columns:
        raise ValueError("add_cross_section_label requires forward_return_pct")
    if sector_col not in frame.columns:
        raise ValueError(f"add_cross_section_label requires '{sector_col}' column")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result[sector_col] = result[sector_col].fillna("Unknown")
    grouped = result.groupby(["date", sector_col], dropna=False)
    result["peer_count"] = grouped["forward_return_pct"].transform("count")
    result["peer_median"] = grouped["forward_return_pct"].transform("median")
    result["cs_excess_pct"] = result["forward_return_pct"] - result["peer_median"]
    result["cs_label"] = result["cs_excess_pct"] > 0
    result["cs_rank_pct"] = grouped["forward_return_pct"].rank(method="average", pct=True)
    result = result[result["peer_count"] >= min_peers].copy()
    return result


def labeled_rows(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
    label_mode: str = "absolute",
) -> pd.DataFrame:
    from tradingai.analysis.ml.features import get_feature_columns

    del label_mode  # single-ticker path always attaches absolute direction first
    labeled = add_direction_label(frame, horizon_days=horizon_days)
    return labeled.dropna(subset=get_feature_columns() + ["direction_up"]).copy()


def labeled_rows_cross_section(
    pooled: pd.DataFrame,
    *,
    min_peers: int = MIN_CS_PEERS,
) -> pd.DataFrame:
    from tradingai.analysis.ml.features import get_feature_columns

    if pooled.empty:
        return pooled.copy()
    labeled = add_cross_section_label(pooled, min_peers=min_peers)
    return labeled.dropna(subset=get_feature_columns() + ["cs_label"]).copy()
