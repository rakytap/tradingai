from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from tradingai.intraday.bounce import assess_bounce
from tradingai.intraday.config import DEFAULT_PROFIT_TAKER_BUFFER
from tradingai.intraday.open_dip import detect_open_dip
from tradingai.intraday.profit_target import target_sell_price
from tradingai.sim.config import (
    DEFAULT_BUY_NOTIONAL_USD,
    DEFAULT_DIP_PCT,
    DEFAULT_DIP_WINDOW_MINUTES,
    DEFAULT_EXCLUDE_FIRST_BAR_GREEN,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_MIN_BOUNCE_SCORE,
    DEFAULT_REQUIRE_STATUS,
    DEFAULT_SIM_MOMENTUM_MINUTES,
    DEFAULT_STOP_LOSS,
    DEFAULT_TARGET_RETURN,
    DEFAULT_USE_TRAIL,
)


@dataclass(frozen=True)
class StrategyParams:
    dip_pct: float = DEFAULT_DIP_PCT
    window_minutes: int = DEFAULT_DIP_WINDOW_MINUTES
    require_status: tuple[str, ...] = DEFAULT_REQUIRE_STATUS
    target_return: float = DEFAULT_TARGET_RETURN
    stop_loss: float = DEFAULT_STOP_LOSS
    sell_trail: float | None = None
    profit_taker_buffer: float = DEFAULT_PROFIT_TAKER_BUFFER
    buy_notional_usd: float = DEFAULT_BUY_NOTIONAL_USD
    max_positions: int = DEFAULT_MAX_POSITIONS
    exclude_first_bar_green: bool = DEFAULT_EXCLUDE_FIRST_BAR_GREEN
    min_bounce_score: float | None = DEFAULT_MIN_BOUNCE_SCORE
    use_trail: bool = DEFAULT_USE_TRAIL
    entry_cutoff_hour: int = 11
    entry_cutoff_minute: int = 0
    momentum_minutes: int = DEFAULT_SIM_MOMENTUM_MINUTES

    def to_dict(self) -> dict:
        return {
            "dip_pct": self.dip_pct,
            "window_minutes": self.window_minutes,
            "require_status": list(self.require_status),
            "target_return": self.target_return,
            "stop_loss": self.stop_loss,
            "sell_trail": self.sell_trail,
            "profit_taker_buffer": self.profit_taker_buffer,
            "buy_notional_usd": self.buy_notional_usd,
            "max_positions": self.max_positions,
            "exclude_first_bar_green": self.exclude_first_bar_green,
            "min_bounce_score": self.min_bounce_score,
            "use_trail": self.use_trail,
            "entry_cutoff_hour": self.entry_cutoff_hour,
            "entry_cutoff_minute": self.entry_cutoff_minute,
            "momentum_minutes": self.momentum_minutes,
        }


@dataclass(frozen=True)
class EntrySignal:
    ticker: str
    session_day: date
    as_of: datetime
    buy_price: float
    dip_pct: float
    status: str
    bounce_score: float
    limit_price: float
    stop_price: float
    trail_activate_price: float | None
    sell_trail: float | None
    profit_target: dict = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DipObservation:
    """Dip/bounce snapshot after a virtual IBKR query (may not be entry-ready)."""

    ticker: str
    as_of: datetime
    qualifies_dip: bool
    dip_pct: float | None
    status: str | None
    bounce_score: float | None
    entry: EntrySignal | None = None

    @property
    def is_plunging_dip(self) -> bool:
        return bool(
            self.qualifies_dip and self.status == "plunging"
        )

    @property
    def is_early_reclaim(self) -> bool:
        """Deep-dip reclaim that just turned positive but score still < 1."""
        return bool(
            self.entry is not None
            and self.status == "reclaiming"
            and self.bounce_score is not None
            and 0.0 < float(self.bounce_score) < 1.0
        )


def entry_rank_key(obs: DipObservation, *, query_index: int = 0) -> tuple:
    """Rank for entry: early reclaim first, then deeper dip, then lower score.

    ``dip_pct`` is negative (deeper = more negative), so ascending sort prefers
    deeper dips. ``query_index`` breaks ties in walk order.
    """
    if obs.entry is None:
        return (9, 0.0, 0.0, query_index, obs.ticker)
    score = float(obs.bounce_score or 0.0)
    dip = float(obs.dip_pct if obs.dip_pct is not None else 0.0)
    if obs.is_early_reclaim:
        tier = 0
        score_part = score  # lower positive = more "just started"
    elif obs.status == "reclaiming":
        tier = 1
        score_part = -score
    else:
        tier = 2
        score_part = 0.0
    return (tier, dip, score_part, query_index, obs.ticker)


class OpenDipStrategy:
    """Deterministic open-dip entry using as-of bars only."""

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or StrategyParams()

    def observe(
        self,
        ticker: str,
        bars_as_of: pd.DataFrame,
        *,
        prior_close: float | None,
        session_day: date,
        as_of: datetime,
    ) -> DipObservation:
        """Assess dip/bounce; build EntrySignal only when entry filters pass."""
        symbol = ticker.upper()
        if bars_as_of is None or bars_as_of.empty:
            return DipObservation(
                ticker=symbol,
                as_of=as_of,
                qualifies_dip=False,
                dip_pct=None,
                status=None,
                bounce_score=None,
            )
        params = self.params
        dip = detect_open_dip(
            bars_as_of,
            ticker=symbol,
            prior_close=prior_close,
            dip_pct=params.dip_pct,
            window_minutes=params.window_minutes,
        )
        if not dip.qualifies:
            return DipObservation(
                ticker=symbol,
                as_of=as_of,
                qualifies_dip=False,
                dip_pct=float(dip.dip_pct) if dip.dip_pct is not None else None,
                status=None,
                bounce_score=None,
            )
        bounce = assess_bounce(
            bars_as_of,
            dip,
            momentum_minutes=params.momentum_minutes,
        )
        entry: EntrySignal | None = None
        buy_price = (
            float(bounce.last_price)
            if bounce.last_price is not None and bounce.last_price > 0
            else None
        )
        if (
            bounce.status in params.require_status
            and not (
                params.exclude_first_bar_green and bounce.first_bar_green
            )
            and (
                params.min_bounce_score is None
                or bounce.bounce_score >= float(params.min_bounce_score)
            )
            and buy_price is not None
        ):
            targets = target_sell_price(
                buy_price,
                buy_notional_usd=params.buy_notional_usd,
                target_return=params.target_return,
                profit_taker_buffer=params.profit_taker_buffer,
                sell_trail=params.sell_trail,
            )
            reasons = list(dict.fromkeys([*dip.reasons, *bounce.reasons]))
            entry = EntrySignal(
                ticker=symbol,
                session_day=session_day,
                as_of=as_of,
                buy_price=buy_price,
                dip_pct=float(dip.dip_pct),
                status=str(bounce.status),
                bounce_score=float(bounce.bounce_score),
                limit_price=float(targets["profit_taker_price"]),
                stop_price=buy_price * (1.0 - float(params.stop_loss)),
                trail_activate_price=(
                    float(targets["trail_activate_price"]) if params.use_trail else None
                ),
                sell_trail=float(targets["sell_trail"]) if params.use_trail else None,
                profit_target=targets,
                reasons=tuple(reasons),
            )
        return DipObservation(
            ticker=symbol,
            as_of=as_of,
            qualifies_dip=True,
            dip_pct=float(dip.dip_pct),
            status=str(bounce.status),
            bounce_score=float(bounce.bounce_score),
            entry=entry,
        )

    def evaluate_entry(
        self,
        ticker: str,
        bars_as_of: pd.DataFrame,
        *,
        prior_close: float | None,
        session_day: date,
        as_of: datetime,
    ) -> EntrySignal | None:
        return self.observe(
            ticker,
            bars_as_of,
            prior_close=prior_close,
            session_day=session_day,
            as_of=as_of,
        ).entry
