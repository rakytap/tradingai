"""Defaults for dip-cut classifier training."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATASET_DIR = Path("data/dip_dataset")
DEFAULT_MODEL_DIR = Path("data/dip_models")

# Sequence: OHLCV channels normalized vs prior_close (+ log volume).
SEQ_CHANNELS = ("open", "high", "low", "close", "volume")
N_SEQ_CHANNELS = len(SEQ_CHANNELS)

# Static head features: log(prior_close), log1p(crossings), minutes/390, side.
N_STATIC_FEATURES = 4

DEFAULT_MAX_LEN = 390
DEFAULT_MIN_BARS = 5
DEFAULT_ALIGN = "right"  # keep bars nearest the cut
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HIDDEN = 64
DEFAULT_BLOCKS = 3
DEFAULT_KERNEL = 5
DEFAULT_DROPOUT = 0.1
DEFAULT_PATIENCE = 8
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 0
