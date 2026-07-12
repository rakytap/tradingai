from tradingai.regime.config import get_fred_api_key, get_regime_providers
from tradingai.regime.models import MacroFeatures
from tradingai.regime.providers.base import MacroDataProvider
from tradingai.regime.providers.fred_macro_provider import FredMacroProvider
from tradingai.regime.providers.yfinance_macro_provider import YFinanceMacroProvider

_PROVIDERS: dict[str, type[MacroDataProvider]] = {
    "yfinance": YFinanceMacroProvider,
    "fred": FredMacroProvider,
}


def _merge_features(base: MacroFeatures, incoming: MacroFeatures) -> MacroFeatures:
    payload = base.model_dump()
    for key, value in incoming.model_dump().items():
        if key == "sources_used":
            continue
        if value is not None:
            payload[key] = value
    payload["sources_used"] = sorted(set(base.sources_used + incoming.sources_used))
    return MacroFeatures(**payload)


def fetch_macro_features() -> MacroFeatures:
    provider_names = get_regime_providers()
    merged: MacroFeatures | None = None

    for name in provider_names:
        if name == "fred" and not get_fred_api_key():
            continue
        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            continue
        try:
            features = provider_cls().fetch_features()
        except Exception:
            continue
        merged = features if merged is None else _merge_features(merged, features)

    if merged is None:
        merged = YFinanceMacroProvider().fetch_features()
    return merged
