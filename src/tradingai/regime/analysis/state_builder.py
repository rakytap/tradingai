from __future__ import annotations

from datetime import datetime, timezone

from tradingai.regime.analysis.classifier import classify_regime, compute_confidence
from tradingai.regime.analysis.feature_builder import build_feature_signals, features_to_dict
from tradingai.regime.analysis.market_factors import classify_market_factors
from tradingai.regime.analysis.sector_board import classify_sector_board
from tradingai.regime.models import MacroFeatures, RegimeState
from tradingai.regime.providers.registry import fetch_macro_features
from tradingai.regime.providers.yfinance_macro_provider import YFinanceMacroProvider
from tradingai.regime.sector_proxies import SECTOR_BOARD, SECTOR_PROXY_TICKERS


def build_regime_state() -> RegimeState:
    """Build structured L1/L4 regime state from live macro + sector ETF series."""
    market_features = fetch_macro_features(trend_symbol="SPY")
    signals = build_feature_signals(market_features)
    market_regime = classify_regime(market_features)
    confidence = compute_confidence(market_features, signals)

    features_by_proxy: dict[str, MacroFeatures] = {}
    for proxy in SECTOR_PROXY_TICKERS:
        try:
            features_by_proxy[proxy] = fetch_macro_features(trend_symbol=proxy)
        except Exception:
            continue

    provider = YFinanceMacroProvider()
    hyg_series = None
    lqd_series = None
    try:
        hyg_series = provider._history("HYG", period="2y")
        lqd_series = provider._history("LQD", period="2y")
    except Exception:
        pass

    xlk = features_by_proxy.get("XLK")
    xlu = features_by_proxy.get("XLU")
    market = classify_market_factors(
        market_features,
        xlk_return_6m_pct=xlk.spy_return_6m_pct if xlk else None,
        xlu_return_6m_pct=xlu.spy_return_6m_pct if xlu else None,
        hyg_series=hyg_series,
        lqd_series=lqd_series,
    )
    sectors = classify_sector_board(features_by_proxy)

    evidence = {
        "market_features": features_to_dict(market_features),
        "market_signals": signals,
        "sector_proxies": dict(SECTOR_BOARD),
        "sector_regimes": {
            item.sector: {"proxy": item.proxy, "regime": item.regime, "stance": item.stance}
            for item in sectors
        },
    }

    return RegimeState(
        market=market,
        sectors=sectors,
        market_regime=market_regime,
        confidence=confidence,
        evidence=evidence,
        as_of=datetime.now(timezone.utc),
    )
