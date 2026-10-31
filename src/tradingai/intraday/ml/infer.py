"""Live inference for the dip-cut recover classifier."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradingai.intraday.dip_dataset import (
    DEFAULT_CROSSING_SESSIONS,
    build_cut,
    count_level_crossings,
    recent_session_window,
    target_sell_price_for_entry,
)
from tradingai.intraday.ml.config import (
    DEFAULT_ALIGN,
    DEFAULT_INFER_MODEL_DIR,
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_BARS,
)
from tradingai.intraday.ml.data import encode_sequence, encode_static
from tradingai.intraday.ml.validate import default_checkpoint_path, load_checkpoint
from tradingai.intraday.ml import LABEL_TO_INDEX


def resolve_dip_checkpoint(path: Path | str | None = None) -> Path | None:
    """Resolve checkpoint path; ``None`` if missing / disabled."""
    raw = (os.getenv("TRADINGAI_DIP_MODEL_CHECKPOINT") or "").strip()
    if path is not None:
        candidate = Path(path)
    elif raw:
        if raw in {"0", "false", "off", "none"}:
            return None
        candidate = Path(raw)
    else:
        candidate = default_checkpoint_path(DEFAULT_INFER_MODEL_DIR)
    if candidate.is_file():
        return candidate
    return None


@dataclass
class DipRecoverPrediction:
    recover_prob: float
    probs: dict[str, float]
    checkpoint: str
    side: str
    sell_crossings_5d: int
    n_bars: int


class DipModelPredictor:
    """Lazy-loaded dip-cut classifier for live open-dip rows."""

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        *,
        device: str = "auto",
    ):
        self._requested = checkpoint
        self._device_name = device
        self._model = None
        self._cfg: dict[str, Any] = {}
        self._checkpoint_path: Path | None = None
        self._load_error: str | None = None
        self._load_attempted = False

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        path = resolve_dip_checkpoint(self._requested)
        if path is None:
            self._load_error = "dip model checkpoint not found / disabled"
            return
        try:
            model, payload = load_checkpoint(path, device=self._device_name)
            self._model = model
            self._cfg = dict(payload.get("config") or {})
            self._checkpoint_path = path
            self._load_error = None
        except Exception as exc:  # pragma: no cover - env/torch issues
            self._model = None
            self._load_error = str(exc)
            self._checkpoint_path = path

    def predict_session(
        self,
        bars: pd.DataFrame,
        *,
        prior_close: float,
        trough_time: pd.Timestamp | None = None,
        cut_end: pd.Timestamp | None = None,
        sell_crossings_5d: float | None = None,
        session_history: list[pd.DataFrame] | None = None,
    ) -> DipRecoverPrediction | None:
        """P(recover / profit target hit) for bars through ``cut_end`` (default last)."""
        self._ensure_loaded()
        if self._model is None:
            return None
        work = bars.sort_index() if bars is not None and not bars.empty else None
        if work is None or work.empty:
            return None
        end = cut_end if cut_end is not None else work.index[-1]
        end = pd.Timestamp(end)
        prior = float(prior_close)
        if not np.isfinite(prior) or prior <= 0:
            return None

        try:
            cut = build_cut(work, prior, end)
        except ValueError:
            return None
        # encode_sequence expects a ``seq`` order column (as in dataset bars).
        cut = cut.copy()
        cut["seq"] = np.arange(len(cut), dtype=np.int64)

        max_len = int(self._cfg.get("max_len", DEFAULT_MAX_LEN))
        min_bars = int(self._cfg.get("min_bars", DEFAULT_MIN_BARS))
        align = str(self._cfg.get("align", DEFAULT_ALIGN))
        if align not in ("right", "left"):
            align = DEFAULT_ALIGN
        n_real = int((~cut["is_prior_close"].astype(bool)).sum())
        if n_real < min_bars:
            return None

        side = "after"
        if trough_time is not None:
            trough = pd.Timestamp(trough_time)
            if end < trough:
                side = "before"

        entry_close = float(work.loc[work.index <= end, "close"].iloc[-1])
        if sell_crossings_5d is None:
            target = target_sell_price_for_entry(entry_close)
            if session_history:
                window = recent_session_window(
                    session_history,
                    end,
                    max_sessions=DEFAULT_CROSSING_SESSIONS,
                )
            else:
                window = work.loc[work.index <= end]
            sell_crossings_5d = float(count_level_crossings(window, target))

        seq, mask = encode_sequence(cut, prior, max_len=max_len, align=align)  # type: ignore[arg-type]
        static = encode_static(
            prior_close=prior,
            sell_crossings_5d=float(sell_crossings_5d),
            cut_end=end,
        )

        import torch

        seq_t = torch.from_numpy(seq).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        static_t = torch.from_numpy(static).unsqueeze(0)
        with torch.no_grad():
            probs = self._model.predict_proba(seq_t, mask_t, static_t)
        arr = probs.detach().cpu().numpy()[0]
        recover_idx = LABEL_TO_INDEX[1]
        recover = float(arr[recover_idx])
        labeled = {
            "-1": round(float(arr[LABEL_TO_INDEX[-1]]), 6),
            "0": round(float(arr[LABEL_TO_INDEX[0]]), 6),
            "1": round(recover, 6),
        }
        return DipRecoverPrediction(
            recover_prob=round(recover, 6),
            probs=labeled,
            checkpoint=str(self._checkpoint_path),
            side=side,
            sell_crossings_5d=int(sell_crossings_5d),
            n_bars=n_real,
        )


_PREDICTOR: DipModelPredictor | None = None


def get_dip_predictor(*, device: str = "auto") -> DipModelPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = DipModelPredictor(device=device)
    return _PREDICTOR


def predict_recover_prob(
    bars: pd.DataFrame,
    *,
    prior_close: float,
    trough_time: pd.Timestamp | None = None,
    cut_end: pd.Timestamp | None = None,
    device: str = "auto",
) -> float | None:
    """Convenience: return P(label=1) or ``None`` if unavailable."""
    pred = get_dip_predictor(device=device).predict_session(
        bars,
        prior_close=prior_close,
        trough_time=trough_time,
        cut_end=cut_end,
    )
    return None if pred is None else pred.recover_prob
