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
    DEFAULT_MAX_PARABOLA_REL_RMSE,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_MIN_BOUNCE_SCORE,
    DEFAULT_REQUIRE_STATUS,
    DEFAULT_SIM_MOMENTUM_MINUTES,
    DEFAULT_STOP_LOSS,
    DEFAULT_TARGET_RETURN,
    DEFAULT_USE_TRAIL,
)


@dataclass(frozen=True)
class AsymmetricParabolaFit:
    """Piecewise fit ``y = a*(x-m)^2 + c`` on each side of the trough."""

    a_left: float
    a_right: float
    trough_price: float
    trough_index: int
    sse: float
    rmse: float
    rel_rmse: float
    depth: float
    n_left: int
    n_right: int


def _fit_half_parabola_a(
    closes: list[float] | tuple[float, ...],
    *,
    trough_pos: int,
    trough_price: float,
    left: bool,
) -> tuple[float, float, int]:
    """Return ``(a, sse, n_points)`` for one arm with fixed vertex ``(trough_pos, c)``."""
    n = len(closes)
    c = float(trough_price)
    num = 0.0
    den = 0.0
    idxs: list[int] = []
    if left:
        idxs = list(range(0, trough_pos + 1))
    else:
        idxs = list(range(trough_pos, n))
    for i in idxs:
        w = float(i - trough_pos) ** 2
        if w <= 1e-18:
            continue
        z = float(closes[i]) - c
        num += w * z
        den += w * w
    if den <= 1e-18:
        return 0.0, float("inf"), len(idxs)
    a = num / den
    sse = 0.0
    for i in idxs:
        w = float(i - trough_pos) ** 2
        pred = c + a * w
        err = float(closes[i]) - pred
        sse += err * err
    return float(a), float(sse), len(idxs)


def fit_asymmetric_parabola(
    bars: pd.DataFrame,
    *,
    prior_close: float | None = None,
) -> AsymmetricParabolaFit | None:
    """Fit left/right half-parabolas on the caller-supplied as-of cut.

    ``bars`` must already be the decision cut (RTH open through decision time).
    The buy is evaluated/placed at the right edge of this series.

    Left:  ``close[i] ≈ a_L*(i-t)^2 + c`` for ``i <= t``
    Right: ``close[i] ≈ a_R*(i-t)^2 + c`` for ``i >= t``
    with ``c = close[t]`` and ``t = argmin(low)`` inside the cut.

    ``rel_rmse = rmse / depth`` where depth is ``max(prior_close - c, max(close)-c, eps)``.
    """
    if bars is None or bars.empty or "close" not in bars.columns:
        return None
    closes = [float(v) for v in bars["close"].astype(float).to_numpy()]
    n = len(closes)
    if n < 3:
        return None
    if "low" in bars.columns:
        trough_pos = int(bars["low"].astype(float).to_numpy().argmin())
    else:
        trough_pos = int(min(range(n), key=lambda i: closes[i]))
    if trough_pos <= 0 or trough_pos >= n - 1:
        return None
    c = closes[trough_pos]
    a_l, sse_l, n_l = _fit_half_parabola_a(
        closes, trough_pos=trough_pos, trough_price=c, left=True
    )
    a_r, sse_r, n_r = _fit_half_parabola_a(
        closes, trough_pos=trough_pos, trough_price=c, left=False
    )
    # U-shape around a minimum requires positive curvature on both arms.
    if a_l <= 1e-18 or a_r <= 1e-18:
        return None
    sse = sse_l + sse_r
    n_pts = max(1, n)
    rmse = (sse / n_pts) ** 0.5
    peak = max(closes)
    depth = max(peak - c, 0.0)
    if prior_close is not None and float(prior_close) > c:
        depth = max(depth, float(prior_close) - c)
    if depth <= 1e-12:
        return None
    return AsymmetricParabolaFit(
        a_left=a_l,
        a_right=a_r,
        trough_price=c,
        trough_index=trough_pos,
        sse=sse,
        rmse=rmse,
        rel_rmse=rmse / depth,
        depth=depth,
        n_left=n_l,
        n_right=n_r,
    )


def asymmetric_parabola_fit_ok(
    bars: pd.DataFrame,
    *,
    prior_close: float | None = None,
    max_parabola_rel_rmse: float = DEFAULT_MAX_PARABOLA_REL_RMSE,
) -> bool:
    """True when the asymmetric parabola fit error is meaningfully small."""
    fit = fit_asymmetric_parabola(bars, prior_close=prior_close)
    if fit is None:
        return False
    return fit.rel_rmse <= max(0.0, float(max_parabola_rel_rmse)) + 1e-12


def asymmetric_parabola_curve(
    bars: pd.DataFrame,
    *,
    prior_close: float | None = None,
) -> tuple[AsymmetricParabolaFit, pd.Series] | None:
    """Return fit and a price series aligned to the as-of cut for plotting."""
    fit = fit_asymmetric_parabola(bars, prior_close=prior_close)
    if fit is None or bars is None or bars.empty:
        return None
    n = len(bars)
    t = int(fit.trough_index)
    c = float(fit.trough_price)
    preds: list[float] = []
    for i in range(n):
        d2 = float(i - t) ** 2
        a = fit.a_left if i <= t else fit.a_right
        preds.append(c + a * d2)
    return fit, pd.Series(preds, index=bars.index, name="parabola")


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
    max_parabola_rel_rmse: float = DEFAULT_MAX_PARABOLA_REL_RMSE

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
            "max_parabola_rel_rmse": self.max_parabola_rel_rmse,
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
        prior = (
            float(dip.prior_close)
            if dip.prior_close is not None
            else (float(prior_close) if prior_close is not None else None)
        )
        # All entry gates use the same as-of cut; buy is at the cut's right edge.
        parabola_ok = asymmetric_parabola_fit_ok(
            bars_as_of,
            prior_close=prior,
            max_parabola_rel_rmse=params.max_parabola_rel_rmse,
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
            and parabola_ok
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
            fit = fit_asymmetric_parabola(bars_as_of, prior_close=prior)
            if fit is not None:
                reasons.append(
                    f"parabola_rel_rmse={fit.rel_rmse:.4f}"
                    f"<= {params.max_parabola_rel_rmse:.4f}"
                    f" a_L={fit.a_left:.6g} a_R={fit.a_right:.6g}"
                )
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
