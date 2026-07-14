from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from tradingai.analysis.ml.artifacts import save_artifact, save_manifest
from tradingai.analysis.ml.config import TRAIN_FRACTION, get_ml_include_regime, get_model_dir
from tradingai.analysis.ml.dataset import (
    build_training_dataset,
    chronological_split,
    split_features_labels,
)
from tradingai.analysis.ml.evaluation import evaluate_predictions, rank_models
from tradingai.analysis.ml.features import get_feature_columns
from tradingai.analysis.ml.registry import get_predictor, list_predictors
from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.stocks import list_tickers
from tradingai.predict.config import get_prediction_horizons


def _resolve_tickers(tickers: list[str] | None) -> list[str]:
    if tickers:
        return [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    registered = list_tickers()
    if registered:
        return registered
    return list(get_sync_tickers())


def train_models(
    *,
    tickers: list[str] | None = None,
    horizons: tuple[int, ...] | None = None,
    train_fraction: float = TRAIN_FRACTION,
    dry_run: bool = False,
    model_dir=None,
) -> dict:
    initialize_schema()
    tickers = _resolve_tickers(tickers)
    horizons = horizons or get_prediction_horizons()
    base_dir = model_dir or get_model_dir()

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "horizons": list(horizons),
        "models": list_predictors(),
        "feature_columns": get_feature_columns(),
        "include_regime": get_ml_include_regime(),
        "results": {},
    }

    for horizon_days in horizons:
        dataset = build_training_dataset(tickers, horizon_days=horizon_days)
        if dataset.empty:
            raise ValueError(
                f"No training rows for horizon {horizon_days}d. "
                "Ensure daily_prices are synced and tickers have enough history."
            )

        train_df, test_df = chronological_split(dataset, train_fraction=train_fraction)
        X_train, y_train = split_features_labels(train_df)
        X_test, y_test = split_features_labels(test_df)

        horizon_metrics: dict[str, dict] = {}
        for model_name in list_predictors():
            predictor = get_predictor(model_name)
            predictor.fit(X_train.to_numpy(), y_train.to_numpy())

            if len(X_test) == 0:
                metrics = {"sample_size": 0, "accuracy": None, "brier_score": None}
            else:
                probabilities = predictor.predict_proba(X_test.to_numpy())
                classes = getattr(predictor, "classes_", None)
                if classes is not None and len(classes) == 2:
                    up_index = int(np.where(classes == 1)[0][0])
                    y_prob = probabilities[:, up_index]
                else:
                    y_prob = probabilities[:, -1]
                metrics = evaluate_predictions(y_test.to_numpy(), y_prob)

            if not dry_run:
                save_artifact(
                    predictor,
                    model_name,
                    horizon_days=horizon_days,
                    model_dir=base_dir,
                )
            horizon_metrics[model_name] = metrics

        manifest["results"][str(horizon_days)] = {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "metrics": horizon_metrics,
            "ranked": rank_models(horizon_metrics),
        }

    if not dry_run:
        save_manifest(manifest, model_dir=base_dir)
    return manifest


def compare_models_from_manifest(*, model_dir=None, horizon_days: int | None = None) -> dict:
    from tradingai.analysis.ml.artifacts import load_manifest

    manifest = load_manifest(model_dir=model_dir or get_model_dir())
    if not manifest:
        raise FileNotFoundError("No manifest.json found. Run tradingai-train-models first.")

    results = manifest.get("results", {})
    if horizon_days is not None:
        key = str(horizon_days)
        if key not in results:
            raise ValueError(f"No metrics for horizon {horizon_days}d in manifest.")
        return {
            "trained_at": manifest.get("trained_at"),
            "horizon_days": horizon_days,
            "ranked": results[key]["ranked"],
            "metrics": results[key]["metrics"],
        }

    comparisons = {}
    for key, payload in results.items():
        comparisons[key] = {
            "ranked": payload["ranked"],
            "metrics": payload["metrics"],
            "train_rows": payload.get("train_rows"),
            "test_rows": payload.get("test_rows"),
        }
    return {
        "trained_at": manifest.get("trained_at"),
        "horizons": comparisons,
    }


def _print_manifest_summary(manifest: dict) -> None:
    for horizon_key, payload in manifest.get("results", {}).items():
        print(f"\nHorizon {horizon_key}d — train={payload['train_rows']} test={payload['test_rows']}")
        for item in payload["ranked"]:
            print(
                f"  #{item['rank']} {item['model']}: "
                f"accuracy={item.get('accuracy')} brier={item.get('brier_score')} "
                f"precision={item.get('precision')} recall={item.get('recall')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train technical analysis ML models.")
    parser.add_argument("--tickers", nargs="*", help="Tickers to include in training.")
    parser.add_argument("--horizon", type=int, help="Train only this horizon in trading days.")
    parser.add_argument("--dry-run", action="store_true", help="Train/evaluate without saving artifacts.")
    args = parser.parse_args()

    horizons = (args.horizon,) if args.horizon else None
    manifest = train_models(tickers=args.tickers, horizons=horizons, dry_run=args.dry_run)
    _print_manifest_summary(manifest)
    if args.dry_run:
        print("\nDry run complete — no artifacts written.")
    else:
        print(f"\nArtifacts saved to {get_model_dir()}")


if __name__ == "__main__":
    main()
