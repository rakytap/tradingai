from __future__ import annotations

from abc import ABC, abstractmethod

from tradingai.regime.models import MacroFeatures


class MacroDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch_features(self) -> MacroFeatures:
        """Fetch macro features contributed by this provider."""
