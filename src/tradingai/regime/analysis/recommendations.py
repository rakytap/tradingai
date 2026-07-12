STRATEGY_MAP = {
    "bull": "increase_trend_exposure",
    "bull_high_volatility": "stay_long_with_hedges",
    "bear": "reduce_long_exposure",
    "high_volatility_bearish": "reduce_long_exposure",
    "sideways": "mean_reversion_or_range_trading",
    "high_volatility": "lower_position_size_and_hedge",
}


def recommend_strategy(regime: str) -> str:
    return STRATEGY_MAP.get(regime, "reduce_position_size_and_reassess")


def explain_regime(regime: str, recommended_strategy: str, signals: list[dict]) -> str:
    highlights = [signal["detail"] for signal in signals[:4]]
    highlight_text = " ".join(highlights)
    disclaimer = (
        "This is macro regime analysis for informational purposes only and is not financial advice."
    )
    return (
        f"Current market regime is {regime}. Recommended strategy: {recommended_strategy}. "
        f"{highlight_text} {disclaimer}"
    )
