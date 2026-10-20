from __future__ import annotations

from pathlib import Path

from tradingai.data.config import (
    DEFAULT_UNIVERSE_FILE,
    get_sync_tickers,
    load_tickers_from_file,
)


def test_packaged_universe_has_about_1000_tickers():
    tickers = load_tickers_from_file(DEFAULT_UNIVERSE_FILE)
    assert len(tickers) == 1000
    assert "AAPL" in tickers
    assert "SPY" not in tickers  # equities file, not ETF bench


def test_get_sync_tickers_uses_file_when_env_unset(monkeypatch):
    monkeypatch.delenv("TRADINGAI_SYNC_TICKERS", raising=False)
    monkeypatch.delenv("TRADINGAI_SYNC_TICKERS_FILE", raising=False)
    get_sync_tickers.cache_clear()
    tickers = get_sync_tickers()
    assert len(tickers) == 1000
    get_sync_tickers.cache_clear()


def test_get_sync_tickers_env_override(monkeypatch):
    monkeypatch.setenv("TRADINGAI_SYNC_TICKERS", "AAA,BBB")
    get_sync_tickers.cache_clear()
    assert get_sync_tickers() == ("AAA", "BBB")
    get_sync_tickers.cache_clear()


def test_load_tickers_from_file_skips_comments(tmp_path: Path):
    path = tmp_path / "u.txt"
    path.write_text("# comment\nAAPL\n\nmsft  # note\nAAPL\nBAD!!\n", encoding="utf-8")
    assert load_tickers_from_file(path) == ("AAPL", "MSFT")
