from __future__ import annotations

import json
from pathlib import Path

import joblib

from tradingai.analysis.ml.config import get_model_dir


def _horizon_dir(model_dir: Path, horizon_days: int) -> Path:
    return model_dir / f"horizon_{horizon_days}"


def artifact_path(
    model_name: str,
    *,
    horizon_days: int,
    model_dir: Path | None = None,
) -> Path:
    base = model_dir or get_model_dir()
    return _horizon_dir(base, horizon_days) / f"{model_name}.joblib"


def manifest_path(model_dir: Path | None = None) -> Path:
    return (model_dir or get_model_dir()) / "manifest.json"


def save_artifact(model, model_name: str, *, horizon_days: int, model_dir: Path | None = None) -> Path:
    path = artifact_path(model_name, horizon_days=horizon_days, model_dir=model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_artifact(
    model_name: str,
    *,
    horizon_days: int,
    model_dir: Path | None = None,
):
    path = artifact_path(model_name, horizon_days=horizon_days, model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"No trained artifact for model '{model_name}' at horizon {horizon_days}d: {path}")
    return joblib.load(path)


def save_manifest(payload: dict, *, model_dir: Path | None = None) -> Path:
    path = manifest_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_manifest(*, model_dir: Path | None = None) -> dict | None:
    path = manifest_path(model_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_manifest_version(*, model_dir: Path | None = None) -> str | None:
    manifest = load_manifest(model_dir=model_dir)
    if not manifest:
        return None
    return manifest.get("trained_at")


def artifact_exists(
    model_name: str,
    *,
    horizon_days: int,
    model_dir: Path | None = None,
) -> bool:
    return artifact_path(model_name, horizon_days=horizon_days, model_dir=model_dir).exists()


def list_available_artifacts(*, model_dir: Path | None = None) -> list[dict]:
    base = model_dir or get_model_dir()
    if not base.exists():
        return []
    available: list[dict] = []
    for horizon_dir in sorted(base.glob("horizon_*")):
        try:
            horizon_days = int(horizon_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        for artifact in sorted(horizon_dir.glob("*.joblib")):
            available.append(
                {
                    "model": artifact.stem,
                    "horizon_days": horizon_days,
                    "path": str(artifact),
                }
            )
    return available
