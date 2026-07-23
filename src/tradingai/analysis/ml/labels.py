from __future__ import annotations

import pandas as pd


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


def labeled_rows(frame: pd.DataFrame, *, horizon_days: int) -> pd.DataFrame:
    from tradingai.analysis.ml.features import get_feature_columns

    labeled = add_direction_label(frame, horizon_days=horizon_days)
    return labeled.dropna(subset=get_feature_columns() + ["direction_up"]).copy()
