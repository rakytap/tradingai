from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from tradingai.analysis.ml.artifacts import save_artifact, save_manifest
from tradingai.analysis.ml.config import (
    MIN_SECTOR_TICKERS,
    TRAIN_FRACTION,
    get_ml_include_regime,
    get_ml_label_mode,
    get_model_dir,
)
from tradingai.analysis.ml.dataset import (
    build_training_dataset,
    chronological_split,
    split_features_labels,
)
from tradingai.analysis.ml.evaluation import evaluate_predictions, rank_models
from tradingai.analysis.ml.features import get_feature_columns
from tradingai.analysis.ml.registry import get_predictor, list_predictors
from tradingai.analysis.ml.sector_models import (
    save_sectors_index,
    sector_model_dir,
    sector_slug,
)
from tradingai.data.config import get_sync_tickers
from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.stocks import list_sectors, list_tickers, list_tickers_by_sector
from tradingai.predict.config import get_prediction_horizons
from tradingai.regime.sector_proxies import normalize_sector_label


def _resolve_tickers(tickers: list[str] | None) -> list[str]:
    if tickers:
        return [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    registered = list_tickers()
    if registered:
        return registered
    return list(get_sync_tickers())


def _resolve_models(models: list[str] | None) -> list[str]:
    if not models:
        return list_predictors()

    resolved: list[str] = []
    for entry in models:
        for part in entry.split(","):
            normalized = part.strip().lower()
            if not normalized:
                continue
            get_predictor(normalized)
            if normalized not in resolved:
                resolved.append(normalized)
    return resolved


def _rank_ic(test_df, y_prob) -> float | None:
    if test_df is None or test_df.empty or "cs_rank_pct" not in test_df.columns:
        return None
    try:
        import pandas as pd

        series_prob = pd.Series(y_prob, index=test_df.index)
        corr = series_prob.corr(test_df["cs_rank_pct"].astype(float), method="spearman")
        if corr != corr:
            return None
        return round(float(corr), 4)
    except Exception:
        return None


def _train_one_universe(
    *,
    tickers: list[str],
    horizons: tuple[int, ...],
    model_names: list[str],
    train_fraction: float,
    dry_run: bool,
    base_dir,
    label_mode: str,
    sector: str | None = None,
    existing_results: dict | None = None,
) -> dict:
    results = dict(existing_results or {})
    for horizon_days in horizons:
        dataset = build_training_dataset(
            tickers,
            horizon_days=horizon_days,
            label_mode=label_mode,
        )
        if dataset.empty:
            raise ValueError(
                f"No training rows for horizon {horizon_days}d"
                + (f" sector={sector}" if sector else "")
                + ". Ensure daily_prices are synced and tickers have enough history."
            )

        train_df, test_df = chronological_split(dataset, train_fraction=train_fraction)
        X_train, y_train = split_features_labels(train_df, label_mode=label_mode)
        X_test, y_test = split_features_labels(test_df, label_mode=label_mode)

        prior_metrics = dict(results.get(str(horizon_days), {}).get("metrics", {}))
        horizon_metrics = {
            name: metrics for name, metrics in prior_metrics.items() if name not in model_names
        }
        for model_name in model_names:
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
                ic = _rank_ic(test_df, y_prob)
                if ic is not None:
                    metrics["rank_ic"] = ic

            if not dry_run:
                save_artifact(
                    predictor,
                    model_name,
                    horizon_days=horizon_days,
                    model_dir=base_dir,
                )
            horizon_metrics[model_name] = metrics

        results[str(horizon_days)] = {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "metrics": horizon_metrics,
            "ranked": rank_models(horizon_metrics),
            "label_mode": label_mode,
            "sector": sector,
        }
    return results


def train_models(
    *,
    tickers: list[str] | None = None,
    horizons: tuple[int, ...] | None = None,
    models: list[str] | None = None,
    train_fraction: float = TRAIN_FRACTION,
    dry_run: bool = False,
    model_dir=None,
    label_mode: str | None = None,
) -> dict:
    initialize_schema()
    tickers = _resolve_tickers(tickers)
    horizons = horizons or get_prediction_horizons()
    model_names = _resolve_models(models)
    base_dir = model_dir or get_model_dir()
    mode = label_mode or get_ml_label_mode()

    from tradingai.analysis.ml.artifacts import load_manifest

    existing_manifest = load_manifest(model_dir=base_dir) or {}

    results = _train_one_universe(
        tickers=tickers,
        horizons=horizons,
        model_names=model_names,
        train_fraction=train_fraction,
        dry_run=dry_run,
        base_dir=base_dir,
        label_mode=mode,
        existing_results=dict(existing_manifest.get("results", {})),
    )

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "horizons": list(horizons),
        "models": list_predictors(),
        "feature_columns": get_feature_columns(),
        "include_regime": get_ml_include_regime(),
        "label_mode": mode,
        "results": results,
    }

    if not dry_run:
        save_manifest(manifest, model_dir=base_dir)
    return manifest


def train_models_by_sector(
    *,
    sectors: list[str] | None = None,
    horizons: tuple[int, ...] | None = None,
    models: list[str] | None = None,
    train_fraction: float = TRAIN_FRACTION,
    dry_run: bool = False,
    model_dir=None,
    label_mode: str | None = None,
    min_sector_tickers: int = MIN_SECTOR_TICKERS,
) -> dict:
    initialize_schema()
    horizons = horizons or get_prediction_horizons()
    model_names = _resolve_models(models)
    base_dir = model_dir or get_model_dir()
    mode = label_mode or get_ml_label_mode()

    available = sectors or list_sectors(exclude_etf=True)
    sector_summaries: dict[str, dict] = {}
    skipped: list[dict] = []

    for sector in available:
        canonical = normalize_sector_label(sector) or sector
        tickers = list_tickers_by_sector(sector, exclude_etf=True)
        if len(tickers) < min_sector_tickers:
            skipped.append(
                {
                    "sector": canonical,
                    "ticker_count": len(tickers),
                    "reason": f"fewer than {min_sector_tickers} tickers",
                }
            )
            continue

        sector_dir = sector_model_dir(canonical, model_dir=base_dir)
        try:
            results = _train_one_universe(
                tickers=tickers,
                horizons=horizons,
                model_names=model_names,
                train_fraction=train_fraction,
                dry_run=dry_run,
                base_dir=sector_dir,
                label_mode=mode,
                sector=canonical,
            )
        except ValueError as exc:
            skipped.append({"sector": canonical, "ticker_count": len(tickers), "reason": str(exc)})
            continue

        manifest = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sector": canonical,
            "sector_slug": sector_slug(canonical),
            "tickers": tickers,
            "horizons": list(horizons),
            "models": list_predictors(),
            "feature_columns": get_feature_columns(),
            "include_regime": get_ml_include_regime(),
            "label_mode": mode,
            "results": results,
        }
        if not dry_run:
            save_manifest(manifest, model_dir=sector_dir)
        sector_summaries[canonical] = {
            "slug": sector_slug(canonical),
            "ticker_count": len(tickers),
            "path": str(sector_dir),
            "results": {
                key: {
                    "train_rows": payload.get("train_rows"),
                    "test_rows": payload.get("test_rows"),
                    "ranked": payload.get("ranked"),
                }
                for key, payload in results.items()
            },
        }

    index = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "label_mode": mode,
        "sectors": sector_summaries,
        "skipped": skipped,
    }
    if not dry_run:
        save_sectors_index(index, model_dir=base_dir)
    return index


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
    if "sectors" in manifest:
        print(f"\nlabel_mode={manifest.get('label_mode')} sectors={len(manifest['sectors'])}")
        for sector, payload in manifest["sectors"].items():
            print(f"\nSector {sector} ({payload['ticker_count']} tickers) → {payload['path']}")
            for horizon_key, horizon_payload in payload.get("results", {}).items():
                print(
                    f"  Horizon {horizon_key}d — "
                    f"train={horizon_payload.get('train_rows')} "
                    f"test={horizon_payload.get('test_rows')}"
                )
                for item in horizon_payload.get("ranked") or []:
                    print(
                        f"    #{item['rank']} {item['model']}: "
                        f"accuracy={item.get('accuracy')} brier={item.get('brier_score')}"
                    )
        if manifest.get("skipped"):
            print("\nSkipped:")
            for item in manifest["skipped"]:
                print(f"  {item['sector']}: {item['reason']}")
        return

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
    parser.add_argument(
        "--model",
        action="append",
        help="Train only this model (repeatable, or comma-separated). "
        "Choices: baseline, logistic, random_forest, xgboost.",
    )
    parser.add_argument(
        "--by-sector",
        action="store_true",
        help="Train one model set per stocks.sector under sectors/<slug>/.",
    )
    parser.add_argument(
        "--cross-section",
        action="store_true",
        help="Use within-sector-date relative labels (above peer median).",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute forward-return direction labels.",
    )
    parser.add_argument("--sector", action="append", help="Limit --by-sector to these sectors.")
    parser.add_argument("--dry-run", action="store_true", help="Train/evaluate without saving artifacts.")
    args = parser.parse_args()

    if args.cross_section and args.absolute:
        raise SystemExit("Use only one of --cross-section or --absolute.")
    label_mode = None
    if args.cross_section:
        label_mode = "cross_section"
    elif args.absolute:
        label_mode = "absolute"

    horizons = (args.horizon,) if args.horizon else None
    if args.by_sector:
        manifest = train_models_by_sector(
            sectors=args.sector,
            horizons=horizons,
            models=args.model,
            dry_run=args.dry_run,
            label_mode=label_mode,
        )
    else:
        manifest = train_models(
            tickers=args.tickers,
            horizons=horizons,
            models=args.model,
            dry_run=args.dry_run,
            label_mode=label_mode,
        )
    _print_manifest_summary(manifest)
    if args.dry_run:
        print("\nDry run complete — no artifacts written.")
    else:
        print(f"\nArtifacts saved under {get_model_dir()}")


if __name__ == "__main__":
    main()
