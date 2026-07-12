from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Quote:
    symbol: str
    price: float
    currency: str | None
    timestamp: datetime | None


class MarketDataProvider(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Return normalized OHLCV data indexed by datetime."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return the latest quote for a symbol."""
