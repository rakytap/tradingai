from __future__ import annotations

from tradingai.regime.models import MacroFeatures


def _trend_label(features: MacroFeatures) -> str:
    if (
        features.spy_close is not None
        and features.spy_sma_200 is not None
        and features.spy_return_6m_pct is not None
        and features.spy_close > features.spy_sma_200
        and features.spy_return_6m_pct > 5
    ):
        return "bull"
    if (
        features.spy_close is not None
        and features.spy_sma_200 is not None
        and features.spy_return_6m_pct is not None
        and features.spy_close < features.spy_sma_200
        and features.spy_return_6m_pct < -5
    ):
        return "bear"
    return "sideways"


def _vol_label(features: MacroFeatures) -> str:
    if features.vix is None:
        return "normal_volatility"
    if features.vix >= 25:
        return "high_volatility"
    if features.vix_percentile_1y is not None and features.vix_percentile_1y >= 75:
        return "high_volatility"
    return "normal_volatility"


def classify_regime(features: MacroFeatures) -> str:
    trend = _trend_label(features)
    vol = _vol_label(features)

    if trend == "bull" and vol == "high_volatility":
        return "bull_high_volatility"
    if trend == "bear" and vol == "high_volatility":
        return "high_volatility_bearish"
    if trend == "sideways" and vol == "high_volatility":
        return "high_volatility"
    return trend


def compute_confidence(features: MacroFeatures, signals: list[dict]) -> float:
    source_count = len(features.sources_used)
    source_component = min(1.0, source_count / 2.0)

    aligned = 0
    total = 0
    trend = next((s for s in signals if s["component"] == "trend"), None)
    vol = next((s for s in signals if s["component"] == "volatility"), None)
    curve = next((s for s in signals if s["component"] == "yield_curve"), None)
    credit = next((s for s in signals if s["component"] == "credit_spreads"), None)

    if trend and vol:
        total += 1
        regime = classify_regime(features)
        if regime.endswith("bearish") or regime == "bear":
            if trend["signal"] == "bear" and vol["signal"] == "high_volatility":
                aligned += 1
        elif regime.startswith("bull"):
            if trend["signal"] == "bull":
                aligned += 1
        elif regime == "sideways":
            if trend["signal"] == "sideways":
                aligned += 1
        elif regime == "high_volatility":
            if vol["signal"] == "high_volatility":
                aligned += 1

    macro_signals = [curve, credit]
    for signal in macro_signals:
        if not signal:
            continue
        total += 1
        if regime := classify_regime(features):
            if "bear" in regime and signal["signal"] in {"inverted", "stressed"}:
                aligned += 1
            elif "bull" in regime and signal["signal"] in {"normal", "rate_context"}:
                aligned += 1

    agreement = aligned / total if total else 0.5
    confidence = 0.45 * source_component + 0.35 * agreement + 0.20
    if "fred" in features.sources_used:
        confidence += 0.1
    return max(0.4, min(0.9, confidence))
