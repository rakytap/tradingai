from __future__ import annotations

from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.regime_snapshots import (
    get_latest_regime_snapshot,
    save_regime_snapshot,
)
from tradingai.regime.analysis.classifier import classify_regime, compute_confidence
from tradingai.regime.analysis.feature_builder import build_feature_signals, features_to_dict
from tradingai.regime.analysis.recommendations import explain_regime, recommend_strategy
from tradingai.regime.analysis.state_builder import build_regime_state
from tradingai.regime.models import RegimeReport, RegimeState
from tradingai.regime.providers.registry import fetch_macro_features


def fetch_macro_features_tool() -> dict:
    features = fetch_macro_features(trend_symbol="SPY")
    signals = build_feature_signals(features)
    return {
        "features": features_to_dict(features),
        "signals": signals,
    }


def detect_market_regime() -> dict:
    features = fetch_macro_features(trend_symbol="SPY")
    signals = build_feature_signals(features)
    regime = classify_regime(features)
    report = RegimeReport(
        regime=regime,
        recommended_strategy=recommend_strategy(regime),
        confidence=compute_confidence(features, signals),
        features=features_to_dict(features),
        signals=signals,
    )
    return report.to_public_dict()


def get_regime_state(*, persist: bool = False) -> dict:
    """Return structured L1/L4 regime state; optionally persist a snapshot."""
    state = build_regime_state()
    payload = state.to_public_dict()
    if persist:
        initialize_schema()
        snapshot_id = save_regime_snapshot(payload, as_of=state.as_of)
        payload["snapshot_id"] = snapshot_id
    return payload


def get_latest_saved_regime_state() -> dict | None:
    initialize_schema()
    row = get_latest_regime_snapshot()
    if row is None:
        return None
    payload = dict(row["payload"])
    payload["snapshot_id"] = row["id"]
    return payload


def load_or_build_regime_state() -> RegimeState:
    """Prefer latest DB snapshot when fresh enough is not required; always rebuild live.

    Fusion needs live sector stances; rebuild each call (provider cache is 15m).
    """
    return build_regime_state()


def explain_market_regime() -> dict:
    features = fetch_macro_features(trend_symbol="SPY")
    signals = build_feature_signals(features)
    regime = classify_regime(features)
    strategy = recommend_strategy(regime)
    return {
        "regime": regime,
        "recommended_strategy": strategy,
        "summary": explain_regime(regime, strategy, signals),
        "signals": signals,
    }
