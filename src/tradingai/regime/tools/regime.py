from __future__ import annotations

from tradingai.regime.analysis.classifier import classify_regime, compute_confidence
from tradingai.regime.analysis.feature_builder import build_feature_signals, features_to_dict
from tradingai.regime.analysis.recommendations import explain_regime, recommend_strategy
from tradingai.regime.models import RegimeReport
from tradingai.regime.providers.registry import fetch_macro_features


def fetch_macro_features_tool() -> dict:
    features = fetch_macro_features()
    signals = build_feature_signals(features)
    return {
        "features": features_to_dict(features),
        "signals": signals,
    }


def detect_market_regime() -> dict:
    features = fetch_macro_features()
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


def explain_market_regime() -> dict:
    features = fetch_macro_features()
    signals = build_feature_signals(features)
    regime = classify_regime(features)
    strategy = recommend_strategy(regime)
    return {
        "regime": regime,
        "recommended_strategy": strategy,
        "summary": explain_regime(regime, strategy, signals),
        "signals": signals,
    }
