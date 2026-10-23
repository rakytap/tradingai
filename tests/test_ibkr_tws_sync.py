from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingai.intraday.config import clear_intraday_config_cache
from tradingai.providers import ibkr_tws


ET = ZoneInfo("America/New_York")


def test_period_to_ib_duration():
    assert ibkr_tws.period_to_ib_duration("5d") == "5 D"
    assert ibkr_tws.period_to_ib_duration("1w") == "1 W"
    with pytest.raises(ValueError):
        ibkr_tws.period_to_ib_duration("5x")


def test_interval_to_ib_bar_size():
    assert ibkr_tws.interval_to_ib_bar_size("1m") == "1 min"
    assert ibkr_tws.interval_to_ib_bar_size("5m") == "5 mins"
    with pytest.raises(ValueError):
        ibkr_tws.interval_to_ib_bar_size("2m")


def test_bars_to_dataframe_normalizes_et():
    bars = [
        SimpleNamespace(
            date=datetime(2026, 10, 15, 9, 30, tzinfo=ET),
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.5,
            volume=1000,
        ),
        SimpleNamespace(
            date=datetime(2026, 10, 15, 9, 31, tzinfo=ET),
            open=100.5,
            high=101.2,
            low=100.0,
            close=100.8,
            volume=800,
        ),
    ]
    frame = ibkr_tws.bars_to_dataframe(bars)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert str(frame.index.tz) == "America/New_York"
    assert len(frame) == 2
    assert float(frame.iloc[0]["open"]) == 100.0


def test_fetch_intraday_ohlcv_ibkr_uses_connection(monkeypatch):
    clear_intraday_config_cache()
    bars = [
        SimpleNamespace(
            date=datetime(2026, 10, 15, 9, 30, tzinfo=ET),
            open=10.0,
            high=11.0,
            low=9.0,
            close=10.5,
            volume=50,
        )
    ]
    calls: dict = {}

    class FakeIB:
        def qualifyContracts(self, contract):
            calls["symbol"] = contract.symbol
            return [contract]

        def reqHistoricalData(self, contract, **kwargs):
            calls["kwargs"] = kwargs
            return bars

    frame = ibkr_tws.fetch_intraday_ohlcv_ibkr(
        FakeIB(),
        "aapl",
        period="5d",
        interval="1m",
    )
    assert calls["symbol"] == "AAPL"
    assert calls["kwargs"]["durationStr"] == "5 D"
    assert calls["kwargs"]["barSizeSetting"] == "1 min"
    assert calls["kwargs"]["endDateTime"] == ""
    assert calls["kwargs"]["useRTH"] is True
    assert len(frame) == 1
    assert float(frame.iloc[0]["close"]) == 10.5


def test_fetch_intraday_ohlcv_ibkr_passes_end_datetime():
    calls: dict = {}

    class FakeIB:
        def reqHistoricalData(self, contract, **kwargs):
            calls["kwargs"] = kwargs
            calls["contract"] = contract
            return [
                SimpleNamespace(
                    date=datetime(2026, 10, 14, 9, 30, tzinfo=ET),
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.0,
                    volume=5,
                )
            ]

    end = datetime(2026, 10, 14, 16, 0, tzinfo=ET)
    ibkr_tws.fetch_intraday_ohlcv_ibkr(
        FakeIB(),
        "AAPL",
        period="1d",
        interval="1m",
        end_datetime=end,
        contract="already-qualified",
    )
    assert calls["contract"] == "already-qualified"
    assert calls["kwargs"]["endDateTime"] == "20261014 20:00:00 UTC"
    assert calls["kwargs"]["durationStr"] == "1 D"


def test_sync_intraday_bars_uses_ibkr(monkeypatch):
    from tradingai.intraday.jobs import sync_intraday_bars as sync_mod

    clear_intraday_config_cache()
    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [10.0],
        },
        index=pd.DatetimeIndex(
            [datetime(2026, 10, 15, 9, 30, tzinfo=ET)],
            name="bar_time",
        ),
    )

    class FakeIB:
        def isConnected(self):
            return True

        def disconnect(self):
            return None

    class FakeCM:
        def __enter__(self):
            return FakeIB()

        def __exit__(self, *args):
            return False

    upserts: list[tuple] = []

    monkeypatch.setattr(sync_mod, "initialize_schema", lambda: None)
    monkeypatch.setattr(
        sync_mod,
        "bullish_sector_watchlist",
        lambda tickers=None, cap=None: ["AAA"],
    )
    monkeypatch.setattr(sync_mod, "connect_ib", lambda: FakeCM())
    monkeypatch.setattr(
        sync_mod,
        "fetch_intraday_ohlcv_ibkr",
        lambda ib, ticker, period=None, interval=None: frame,
    )
    monkeypatch.setattr(
        sync_mod,
        "upsert_intraday_bars",
        lambda ticker, rows: upserts.append((ticker, rows)) or len(rows),
    )
    monkeypatch.setattr(sync_mod, "prune_intraday_bars", lambda keep_days=5: 0)
    monkeypatch.setattr(sync_mod, "log_start", lambda *a, **k: None)
    monkeypatch.setattr(sync_mod, "log_progress", lambda *a, **k: None)
    monkeypatch.setattr(sync_mod, "log_error", lambda *a, **k: None)
    monkeypatch.setattr(sync_mod, "log_done", lambda *a, **k: None)
    monkeypatch.setattr(sync_mod, "get_ibkr_pace_seconds", lambda: 0.0)

    result = sync_mod.sync_intraday_bars(cap=1)
    assert result["source"] == "ibkr_tws"
    assert result["synced"] == 1
    assert result["bars_written"] == 1
    assert upserts[0][0] == "AAA"
    assert len(upserts[0][1]) == 1


def test_sync_connection_failure_raises(monkeypatch):
    from tradingai.intraday.jobs import sync_intraday_bars as sync_mod

    clear_intraday_config_cache()

    class FailCM:
        def __enter__(self):
            raise ConnectionRefusedError("refused")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(sync_mod, "initialize_schema", lambda: None)
    monkeypatch.setattr(
        sync_mod,
        "bullish_sector_watchlist",
        lambda tickers=None, cap=None: ["AAA"],
    )
    monkeypatch.setattr(sync_mod, "connect_ib", lambda: FailCM())
    monkeypatch.setattr(sync_mod, "log_start", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="IBKR TWS connection failed"):
        sync_mod.sync_intraday_bars(cap=1)
