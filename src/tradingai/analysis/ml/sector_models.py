from __future__ import annotations

import json
import re
from pathlib import Path

from tradingai.analysis.ml.artifacts import artifact_exists
from tradingai.analysis.ml.config import get_ml_allow_global_fallback, get_model_dir
from tradingai.regime.sector_proxies import normalize_sector_label

MIN_SECTOR_TICKERS = 8


def sector_slug(sector: str | None) -> str:
    label = normalize_sector_label(sector) or (sector or "unknown")
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return slug.strip("_") or "unknown"


def sectors_root(model_dir: Path | None = None) -> Path:
    return (model_dir or get_model_dir()) / "sectors"


def sector_model_dir(sector: str | None, *, model_dir: Path | None = None) -> Path:
    return sectors_root(model_dir) / sector_slug(sector)


def sectors_index_path(model_dir: Path | None = None) -> Path:
    return sectors_root(model_dir) / "index.json"


def save_sectors_index(payload: dict, *, model_dir: Path | None = None) -> Path:
    path = sectors_index_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_sectors_index(*, model_dir: Path | None = None) -> dict | None:
    path = sectors_index_path(model_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_model_dir_for_sector(
    sector: str | None,
    *,
    model_name: str,
    horizon_days: int,
    model_dir: Path | None = None,
    allow_global_fallback: bool | None = None,
) -> tuple[Path, str]:
    """Return (artifact_root, source) where source is 'sector' or 'global'."""
    base = model_dir or get_model_dir()
    sector_dir = sector_model_dir(sector, model_dir=base)
    if artifact_exists(model_name, horizon_days=horizon_days, model_dir=sector_dir):
        return sector_dir, "sector"

    allow = (
        get_ml_allow_global_fallback()
        if allow_global_fallback is None
        else allow_global_fallback
    )
    if allow and artifact_exists(model_name, horizon_days=horizon_days, model_dir=base):
        return base, "global"

    raise FileNotFoundError(
        f"No sector artifact for '{normalize_sector_label(sector) or sector}' "
        f"(model={model_name}, horizon={horizon_days}d) under {sector_dir}"
        + (" and global fallback disabled" if not allow else " and no global artifact")
    )
