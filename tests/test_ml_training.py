import numpy as np
import pandas as pd

from tradingai.analysis.ml.artifacts import load_manifest, manifest_path
from tradingai.analysis.ml.evaluation import evaluate_predictions, rank_models
from tradingai.analysis.ml.training import _resolve_tickers, train_models


def _training_frame(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D").date
    data = {
        "date": dates,
        "ticker": ["AAPL"] * rows,
        "close": np.linspace(100, 150, rows),
        "direction_up": rng.integers(0, 2, size=rows).astype(bool),
    }
    from tradingai.analysis.ml.features import FEATURE_COLUMNS

    for column in FEATURE_COLUMNS:
        data[column] = rng.normal(size=rows)
    return pd.DataFrame(data)


def test_resolve_tickers_uses_string_list_from_database(monkeypatch):
    monkeypatch.setattr(
        "tradingai.analysis.ml.training.list_tickers",
        lambda: ["AAPL", "MSFT"],
    )
    assert _resolve_tickers(None) == ["AAPL", "MSFT"]


def test_train_models_writes_manifest_and_artifacts(tmp_path, monkeypatch):
    frame = _training_frame()

    monkeypatch.setattr(
        "tradingai.analysis.ml.training.build_training_dataset",
        lambda tickers, horizon_days: frame.copy(),
    )
    monkeypatch.setattr("tradingai.analysis.ml.training.initialize_schema", lambda: None)

    manifest = train_models(tickers=["AAPL"], horizons=(5,), model_dir=tmp_path)
    assert manifest_path(tmp_path).exists()
    loaded = load_manifest(model_dir=tmp_path)
    assert loaded is not None
    assert "5" in loaded["results"]
    assert (tmp_path / "horizon_5" / "logistic.joblib").exists()


def test_rank_models_orders_by_brier_score():
    metrics = {
        "baseline": {"brier_score": 0.30, "accuracy": 0.52},
        "logistic": {"brier_score": 0.22, "accuracy": 0.58},
    }
    ranked = rank_models(metrics)
    assert ranked[0]["model"] == "logistic"
    assert ranked[0]["rank"] == 1


def test_evaluate_predictions_returns_core_metrics():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.6, 0.7])
    metrics = evaluate_predictions(y_true, y_prob)
    assert metrics["sample_size"] == 4
    assert metrics["accuracy"] is not None
    assert metrics["brier_score"] is not None
