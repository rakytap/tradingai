import json

from tradingai.analysis.ml.training import compare_models_from_manifest


def test_compare_models_from_manifest_all_horizons(tmp_path):
    manifest = {
        "trained_at": "2026-01-01T00:00:00+00:00",
        "results": {
            "5": {
                "train_rows": 100,
                "test_rows": 25,
                "metrics": {
                    "baseline": {"accuracy": 0.5, "brier_score": 0.28},
                    "logistic": {"accuracy": 0.6, "brier_score": 0.22},
                },
                "ranked": [
                    {"model": "logistic", "rank": 1, "accuracy": 0.6, "brier_score": 0.22},
                    {"model": "baseline", "rank": 2, "accuracy": 0.5, "brier_score": 0.28},
                ],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    result = compare_models_from_manifest(model_dir=tmp_path)
    assert result["trained_at"] == "2026-01-01T00:00:00+00:00"
    assert "5" in result["horizons"]
    assert result["horizons"]["5"]["ranked"][0]["model"] == "logistic"


def test_compare_models_from_manifest_single_horizon(tmp_path):
    manifest = {
        "trained_at": "2026-01-01T00:00:00+00:00",
        "results": {
            "5": {
                "metrics": {"logistic": {"accuracy": 0.6, "brier_score": 0.22}},
                "ranked": [{"model": "logistic", "rank": 1, "accuracy": 0.6, "brier_score": 0.22}],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    result = compare_models_from_manifest(model_dir=tmp_path, horizon_days=5)
    assert result["horizon_days"] == 5
    assert result["ranked"][0]["model"] == "logistic"
