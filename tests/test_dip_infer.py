"""Live dip-model inference for open-dip / dipwalk Prob column."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from types import SimpleNamespace

from tradingai.intraday.ml.config import DEFAULT_ALIGN, DEFAULT_MAX_LEN, DEFAULT_MIN_BARS
from tradingai.intraday.ml.infer import DipModelPredictor, resolve_dip_checkpoint
from tradingai.intraday.ml.model import DipCutClassifier
from tradingai.intraday.scan import _row_payload

ET = ZoneInfo("America/New_York")


def _session_bars(n: int = 40) -> pd.DataFrame:
    start = datetime(2026, 10, 28, 9, 30, tzinfo=ET)
    index = pd.date_range(start, periods=n, freq="1min", tz=ET)
    close = 100.0 * (0.95 - 0.0005 * np.arange(n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=index,
    )


def _tiny_checkpoint(tmp_path: Path) -> Path:
    import torch

    model = DipCutClassifier(hidden=8, blocks=1, kernel=3, dropout=0.0)
    out = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": "tradingai_dip_model",
            "epoch": 1,
            "best_f1": 0.0,
            "config": {
                "max_len": DEFAULT_MAX_LEN,
                "min_bars": DEFAULT_MIN_BARS,
                "align": DEFAULT_ALIGN,
                "hidden": 8,
                "blocks": 1,
                "kernel": 3,
                "dropout": 0.0,
            },
            "best_model_state": model.state_dict(),
            "model_state": model.state_dict(),
        },
        out,
    )
    return out


def test_resolve_checkpoint_off(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAI_DIP_MODEL_CHECKPOINT", "off")
    assert resolve_dip_checkpoint() is None
    monkeypatch.setenv("TRADINGAI_DIP_MODEL_CHECKPOINT", str(tmp_path / "missing.pt"))
    assert resolve_dip_checkpoint() is None


def test_predictor_recover_prob(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path)
    pred = DipModelPredictor(ckpt, device="cpu")
    bars = _session_bars()
    out = pred.predict_session(
        bars,
        prior_close=100.0,
        trough_time=bars.index[10],
    )
    assert out is not None
    assert 0.0 <= out.recover_prob <= 1.0
    assert abs(sum(out.probs.values()) - 1.0) < 1e-5
    assert "1" in out.probs


def test_row_payload_attaches_recover_prob(tmp_path, monkeypatch):
    ckpt = _tiny_checkpoint(tmp_path)
    monkeypatch.setenv("TRADINGAI_DIP_MODEL_CHECKPOINT", str(ckpt))
    # Reset singleton so env is picked up.
    import tradingai.intraday.ml.infer as infer_mod

    infer_mod._PREDICTOR = None

    bars = _session_bars()
    dip = SimpleNamespace(
        ticker="AAA",
        dip_pct=-5.0,
        trough_price=float(bars["low"].min()),
        prior_close=100.0,
        session_open=float(bars["open"].iloc[0]),
        reasons=("dip",),
        trough_time=bars.index[12],
    )
    bounce = SimpleNamespace(
        status="reclaiming",
        bounce_score=0.5,
        last_price=float(bars["close"].iloc[-1]),
        vwap=None,
        reclaim_vwap=False,
        reclaim_open=False,
        first_bar_green=False,
        trough_hammer=False,
        bullish_engulfing=False,
        bearish_engulfing=False,
        morning_star=False,
        momentum_15m_pct=0.0,
        reasons=(),
    )
    row = _row_payload(
        dip,
        bounce,
        buy_notional_usd=20000,
        target_return=0.0038,
        sell_trail=0.0019,
        profit_taker_buffer=0.0002,
        bars=bars,
    )
    assert "recover_prob" in row
    assert 0.0 <= row["recover_prob"] <= 1.0
