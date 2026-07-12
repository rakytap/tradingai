from tradingai.predict.models import AgentSignal, HorizonPrediction, PredictionReport
from tradingai.predict.tagging.prediction_tagger import build_prediction_metadata


def _report(
    *,
    technical: float = 0.8,
    news_score: float = 0.3,
    catalysts: list[str] | None = None,
    regime: str = "bull",
) -> PredictionReport:
    return PredictionReport(
        ticker="NVDA",
        agent_signals=[
            AgentSignal(
                agent="technical",
                ticker="NVDA",
                directional_score=technical,
                bullish_score=0.9,
                bearish_score=0.1,
                confidence=0.7,
                reasons=["Strong momentum"],
                raw={},
            ),
            AgentSignal(
                agent="news",
                ticker="NVDA",
                directional_score=news_score,
                bullish_score=0.65,
                bearish_score=0.35,
                confidence=0.8,
                reasons=[],
                raw={
                    "catalysts": catalysts or ["beat estimates on revenue growth"],
                    "risk_factors": ["valuation"],
                },
            ),
            AgentSignal(
                agent="regime",
                ticker=None,
                directional_score=0.8,
                bullish_score=0.9,
                bearish_score=0.1,
                confidence=0.7,
                reasons=[],
                raw={"regime": regime},
            ),
        ],
        fusion={
            "technical_score": technical,
            "news_score": news_score,
            "regime_score": 0.8,
            "combined_score": 0.7,
        },
        horizons=[
            HorizonPrediction(horizon_days=5, probability_up=0.68, expected_return_pct=1.2)
        ],
        confidence=0.8,
        reasons=["Strong momentum"],
    )


def test_earnings_driven_rally_classification():
    metadata = build_prediction_metadata(
        _report(catalysts=["Company beat estimates on earnings"]),
        sector="Technology",
    )
    assert metadata.prediction_type == "earnings_driven_rally"
    assert "earnings_surprise" in metadata.catalyst_tags


def test_product_catalyst_classification():
    metadata = build_prediction_metadata(
        _report(catalysts=["New product rollout announced"]),
        sector="Technology",
    )
    assert metadata.prediction_type == "product_catalyst_rally"


def test_risk_off_drawdown_classification():
    report = _report(technical=-0.5, news_score=-0.4, regime="bear")
    report.fusion["combined_score"] = -0.4
    metadata = build_prediction_metadata(report, sector="Technology")
    assert metadata.prediction_type == "risk_off_drawdown"
