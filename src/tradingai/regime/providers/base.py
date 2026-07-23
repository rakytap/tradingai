from __future__ import annotations

from abc import ABC, abstractmethod

from tradingai.regime.models import MacroFeatures


class MacroDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch_features(self, *, trend_symbol: str = "SPY") -> MacroFeatures:
        """Fetch macro features contributed by this provider.

        ``trend_symbol`` selects the equity series used for SMA / 6m trend.
        Providers that do not supply trend series may ignore it.
        """
