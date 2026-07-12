from datetime import datetime, timezone

from tradingai.predict.evaluation.metrics import compute_evaluation_metrics
from tradingai.predict.fusion.scorer import fuse_agent_signals
from tradingai.predict.models import AgentSignal
from tradingai.predict.storage.repository import save_outcome, save_prediction


def _report():
    signals = [
        AgentSignal(
            agent="technical",
            ticker="AAPL",
            directional_score=0.5,
            bullish_score=0.75,
            bearish_score=0.25,
            confidence=0.6,
            reasons=["Bullish technical setup"],
            raw={"bias": "bullish"},
        ),
        AgentSignal(
            agent="news",
            ticker="AAPL",
            directional_score=0.2,
            bullish_score=0.6,
            bearish_score=0.4,
            confidence=0.7,
            reasons=["Positive catalyst"],
            raw={"sentiment": 0.6},
        ),
        AgentSignal(
            agent="regime",
            ticker=None,
            directional_score=0.8,
            bullish_score=0.9,
            bearish_score=0.1,
            confidence=0.75,
            reasons=["Bull regime"],
            raw={"regime": "bull"},
        ),
    ]
    report = fuse_agent_signals("AAPL", signals)
    report.as_of = datetime(2026, 1, 10, tzinfo=timezone.utc)
    return report


def test_compute_evaluation_metrics(tmp_path):
    db_path = str(tmp_path / "predictions.sqlite3")
    report = _report()
    prediction_id = save_prediction(report, db_path=db_path)
    save_outcome(prediction_id, 5, 2.5, db_path=db_path)

    metrics = compute_evaluation_metrics(5, db_path=db_path)
    horizon = metrics["horizons"]["5"]
    assert horizon["sample_size"] == 1
    assert horizon["direction_accuracy"] == 1.0
    assert horizon["precision"] == 1.0
    assert horizon["recall"] == 1.0
    assert horizon["brier_score"] is not None
    assert horizon["mae_expected_return_pct"] is not None

    rows = []
    for _ in range(12):
        report.as_of = report.as_of.replace(day=10)
        pid = save_prediction(report, db_path=db_path)
        save_outcome(pid, 5, 1.0, db_path=db_path)
        rows.append(pid)
    assert rows

    metrics = compute_evaluation_metrics(5, db_path=db_path)
    assert metrics["horizons"]["5"]["sample_size"] >= 2
    assert metrics["horizons"]["5"]["top_10_ranked_return_pct"] is not None
