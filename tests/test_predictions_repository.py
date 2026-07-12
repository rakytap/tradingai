from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from tradingai.data.repositories import predictions as predictions_repo


def test_get_track_record_query():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (10, 0.54, 0.9, 0.62)

    with patch("tradingai.data.repositories.predictions.get_connection") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_conn
        result = predictions_repo.get_track_record(
            sector="semiconductor",
            prediction_type="earnings_driven_rally",
            horizon_days=5,
        )

    assert result["sample_size"] == 10
    assert result["hit_rate"] == 0.54
    assert result["overconfidence"] == 0.08


def test_save_prediction_rows_returns_ids():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.side_effect = [(1,), (2,), (3,)]

    rows = [
        {
            "ticker": "AAPL",
            "timestamp": datetime(2026, 1, 10, tzinfo=timezone.utc),
            "horizon_days": 5,
            "probability_up": 0.68,
            "expected_return_pct": 1.2,
            "confidence": 0.7,
            "combined_score": 0.6,
            "catalyst_tags": ["earnings_surprise"],
            "reason_tags": [],
            "prediction_type": "earnings_driven_rally",
            "agent_payload": {},
            "reasons": ["test"],
        }
    ]

    with patch("tradingai.data.repositories.predictions.get_connection") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_conn
        ids = predictions_repo.save_prediction_rows(rows)

    assert ids == [1]
