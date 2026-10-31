"""Training cuts around the most pronounced morning valley.

Within the first ``valley_window_minutes`` of the RTH session (default 90),
find the most pronounced valley on 1-minute lows, ranked by **right-side**
bounce (left shoulder may be tiny, e.g. an open washout). The valley bottom
is the dip anchor — not the session-wide low and not a cut end. Cuts are
``n_cuts`` (default 8) random 1-minute bars drawn from
``[session_open + cut_min_minutes_from_open,
session_open + cut_max_minutes_from_open]`` (default 60–150 min = 1.0–2.5h),
excluding the anchor itself. Side (before/after the dip) is derived from
each cut for labeling. The valley must reach at least ``dip_pct`` below the
prior close.

The label (after-dip cuts judged bar-by-bar within ``label_horizon_minutes``,
default 15):

* ``-1`` if the cut is on the plunging side of the valley (strictly before
  the dip).
* ``1`` if the fee-aware sell target is reached **before** the stop (any high
  ≥ target with no earlier stop touch).
* ``0`` if the stop is hit first (low ≤ cut×(1−stop_loss), default 0.6%), or
  the target is never reached in the horizon. Same-bar dual touch → ``0``.
* ``sell_crossings_5d``: how many times 1m closes crossed *up through* the
  fee-aware target sell price (from ``entry_close`` /
  ``TRADINGAI_OPEN_DIP_TARGET_RETURN`` / ``get_target_return()``)
  over the last five RTH sessions through the cut (no post-cut lookahead).
  Downward crosses do not count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from collections import defaultdict, deque

from tradingai.intraday.bars import filter_rth_session
from tradingai.intraday.config import (
    DEFAULT_DIP_LABEL_HORIZON_MINUTES,
    DEFAULT_DIP_LABEL_STOP_LOSS,
    get_buy_notional_usd,
    get_label_horizon_minutes,
    get_label_stop_loss,
    get_target_return,
)
from tradingai.intraday.profit_target import fee_aware_sell_price

DEFAULT_DATASET_DIP_PCT = 4.0
DEFAULT_DATASET_RISE_PCT = 0.4  # retained in sample rows for compatibility
DEFAULT_CUTS_PER_SESSION = 8
# Deprecated aliases kept for older call sites / imports.
DEFAULT_CUTS_BEFORE = 4
DEFAULT_CUTS_AFTER = 4
DEFAULT_LABEL_HORIZON_MINUTES = DEFAULT_DIP_LABEL_HORIZON_MINUTES
DEFAULT_MIN_HORIZON_MINUTES = DEFAULT_LABEL_HORIZON_MINUTES
DEFAULT_CUT_VICINITY_MINUTES = 0  # unused; kept for CLI/import compat
# Session-relative cut window (minutes after first RTH bar / session open).
DEFAULT_CUT_MIN_MINUTES_FROM_OPEN = 60  # 1.0 hour
DEFAULT_CUT_MAX_MINUTES_FROM_OPEN = 150  # 2.5 hours
DEFAULT_VALLEY_WINDOW_MINUTES = 90  # first 1.5 RTH hours
DEFAULT_VALLEY_ORDER_MINUTES = 5  # local-min half-width
DEFAULT_CROSSING_SESSIONS = 5
# Post-dip stop vs cut close: any low ≤ entry*(1-stop_loss) → label 0.
# Overridable via TRADINGAI_DIP_LABEL_STOP_LOSS / get_label_stop_loss().
DEFAULT_LABEL_STOP_LOSS = DEFAULT_DIP_LABEL_STOP_LOSS

SAMPLE_COLUMNS = (
    "sample_id",
    "ticker",
    "session_date",
    "cut_end",
    "side",
    "prior_close",
    "entry_close",
    "anchor_time",
    "dip_pct",
    "max_future_high",
    "rise_pct",
    "label",
    "target_sell_price",
    "stop_loss_price",
    "sell_crossings_5d",
)
BAR_COLUMNS = (
    "sample_id",
    "seq",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_prior_close",
)


@dataclass(frozen=True)
class DipAnchor:
    anchor_time: pd.Timestamp
    anchor_low: float
    dip_pct: float


def _sorted_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    work = bars.sort_index()
    if work.index.has_duplicates:
        work = work[~work.index.duplicated(keep="last")]
    return work


def _as_index_timestamp(value: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if index.tz is None:
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("America/New_York").tz_localize(None)
        return stamp
    if stamp.tzinfo is None:
        return stamp.tz_localize(index.tz)
    return stamp.tz_convert(index.tz)


def _morning_window(
    bars: pd.DataFrame,
    *,
    valley_window_minutes: int = DEFAULT_VALLEY_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Bars from the session open through the first ``valley_window_minutes``."""
    work = _sorted_bars(bars)
    if work.empty:
        return work
    minutes = max(1, int(valley_window_minutes))
    start = pd.Timestamp(work.index[0])
    end = start + pd.Timedelta(minutes=minutes)
    # Inclusive of the bar that opens at ``end`` if present (e.g. 11:00 for 90m).
    return work.loc[work.index <= end]


def _valley_prominence(lows: np.ndarray, index: int) -> float:
    """Right-shoulder rise after the valley bottom (left side may be tiny).

    Score is ``max(lows[index:]) - lows[index]``. The left shoulder is not part
    of the ranking — open washouts with little left structure can still win if
    they bounce hard to the right.
    """
    right = lows[index:]
    if right.size < 2:
        return 0.0
    right_peak = float(np.nanmax(right))
    bottom = float(lows[index])
    if not math.isfinite(right_peak) or not math.isfinite(bottom):
        return 0.0
    return float(right_peak - bottom)


def find_dip_anchor(
    bars: pd.DataFrame,
    prior_close: float,
    dip_pct: float = DEFAULT_DATASET_DIP_PCT,
    *,
    valley_window_minutes: int = DEFAULT_VALLEY_WINDOW_MINUTES,
    valley_order_minutes: int = DEFAULT_VALLEY_ORDER_MINUTES,
) -> DipAnchor | None:
    """Most pronounced valley bottom in the first ``valley_window_minutes``.

    A local minimum on 1-minute lows is scored by **right-side** recovery
    inside the morning window (how far lows bounce after the trough). The left
    shoulder may be only a few bars. Candidates need ``valley_order_minutes`` of
    bars on the right; the left side only needs at least one prior bar. The
    highest right-rise valley whose low is at least ``dip_pct`` under
    ``prior_close`` becomes the dip. Post-window bars are never used.

    ``dip_pct`` is a positive percent (4 means −4%). The returned ``dip_pct`` is
    the valley depth vs prior close, in percent, and is negative.
    """
    if prior_close is None or not math.isfinite(prior_close) or prior_close <= 0:
        return None
    morning = _morning_window(bars, valley_window_minutes=valley_window_minutes)
    if morning.empty or "low" not in morning.columns:
        return None
    lows = morning["low"].astype(float).to_numpy()
    n = int(lows.size)
    order = max(1, int(valley_order_minutes))
    # Need ≥1 bar on the left and ``order`` bars on the right.
    if n < order + 2:
        return None

    threshold = float(prior_close) * (1.0 - abs(float(dip_pct)) / 100.0)
    best_i: int | None = None
    best_prom = -1.0
    best_low = math.inf
    for i in range(1, n - order):
        value = float(lows[i])
        if not math.isfinite(value) or value > threshold + 1e-9:
            continue
        # Local min with asymmetric neighborhood: full ``order`` on the right,
        # only the available left span (may be smaller than ``order``).
        left_span = min(order, i)
        neighborhood = lows[i - left_span : i + order + 1]
        if float(np.nanmin(neighborhood)) < value - 1e-12:
            continue
        if np.any(lows[i - left_span : i] <= value + 1e-12):
            continue
        if value > float(lows[i - 1]) + 1e-12 or value > float(lows[i + 1]) + 1e-12:
            continue
        prom = _valley_prominence(lows, i)
        if prom <= 0.0:
            continue
        if (
            prom > best_prom + 1e-12
            or (abs(prom - best_prom) <= 1e-12 and value < best_low - 1e-12)
        ):
            best_i = i
            best_prom = prom
            best_low = value

    if best_i is None:
        return None
    anchor_time = pd.Timestamp(morning.index[best_i])
    anchor_low = float(lows[best_i])
    depth = round((anchor_low / float(prior_close) - 1.0) * 100.0, 4)
    return DipAnchor(
        anchor_time=anchor_time,
        anchor_low=round(anchor_low, 6),
        dip_pct=depth,
    )


def _take_positions(pool: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or not pool:
        return []
    chosen_count = min(int(count), len(pool))
    chosen = np.atleast_1d(
        rng.choice(np.asarray(pool, dtype=int), size=chosen_count, replace=False)
    )
    return sorted(int(item) for item in chosen)


def sample_cut_ends(
    bars: pd.DataFrame,
    anchor_time: pd.Timestamp,
    n_cuts: int,
    min_horizon_minutes: int,
    rng: np.random.Generator,
    *,
    cut_min_minutes_from_open: int = DEFAULT_CUT_MIN_MINUTES_FROM_OPEN,
    cut_max_minutes_from_open: int = DEFAULT_CUT_MAX_MINUTES_FROM_OPEN,
    # Deprecated kwargs (ignored) — kept so older call sites do not break.
    n_before: int | None = None,
    n_after: int | None = None,
    vicinity_minutes: int | None = None,
) -> list[tuple[pd.Timestamp, str]]:
    """Random cut ends in the session window around the dip.

    Eligible bars:
    * clock in
      ``[session_open + cut_min_minutes_from_open,
      session_open + cut_max_minutes_from_open]`` (session open = first bar),
    * not equal to the dip ``anchor_time``,
    * at least ``min_horizon_minutes`` later bars remain.

    Up to ``n_cuts`` bars are sampled without replacement. Each cut is tagged
    ``before`` / ``after`` relative to the dip for labeling. If
    ``n_before``/``n_after`` are passed (legacy), ``n_cuts`` defaults to their
    sum when ``n_cuts`` is omitted via the positional API — callers should
    pass ``n_cuts`` explicitly.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    # Legacy: sample_cut_ends(bars, anchor, n_before, n_after, horizon, rng)
    # was the old signature. Detect mistaken positional use via n_before when
    # the 3rd arg was actually n_before and 4th n_after — handled by callers
    # that still pass n_before=/n_after= keywords.
    if n_before is not None or n_after is not None:
        legacy = int(n_before or 0) + int(n_after or 0)
        if legacy > 0:
            n_cuts = legacy
    _ = vicinity_minutes  # legacy no-op
    work = _sorted_bars(bars)
    if work.empty:
        return []
    count = max(0, int(n_cuts))
    if count <= 0:
        return []
    horizon = max(0, int(min_horizon_minutes))
    last_eligible = len(work) - horizon - 1
    if last_eligible < 0:
        return []
    anchor = _as_index_timestamp(anchor_time, work.index)
    session_open = pd.Timestamp(work.index[0])
    min_from_open = int(cut_min_minutes_from_open)
    max_from_open = int(cut_max_minutes_from_open)
    earliest = (
        session_open + pd.Timedelta(minutes=min_from_open)
        if min_from_open > 0
        else session_open
    )
    latest = (
        session_open + pd.Timedelta(minutes=max_from_open)
        if max_from_open > 0
        else pd.Timestamp(work.index[last_eligible])
    )
    if earliest > latest:
        raise ValueError(
            "cut_min_minutes_from_open must be <= cut_max_minutes_from_open "
            "when both are positive."
        )
    pool: list[int] = []
    for position in range(last_eligible + 1):
        stamp = pd.Timestamp(work.index[position])
        if stamp < earliest or stamp > latest:
            continue
        if stamp == anchor:
            continue
        pool.append(position)
    chosen: list[tuple[pd.Timestamp, str]] = []
    for position in _take_positions(pool, count, rng):
        stamp = pd.Timestamp(work.index[position])
        side = "before" if stamp < anchor else "after"
        chosen.append((stamp, side))
    return chosen


def build_cut(
    bars: pd.DataFrame,
    prior_close: float,
    cut_end: pd.Timestamp,
) -> pd.DataFrame:
    """Prior-close row plus 1-minute bars from the open through ``cut_end``.

    The prior-close timestamp is one minute before the session's first bar.
    It is a sentinel, marked ``is_prior_close``, not the prior session clock time.
    """
    work = _sorted_bars(bars)
    end = _as_index_timestamp(cut_end, work.index)
    window = work.loc[work.index <= end, ["open", "high", "low", "close", "volume"]].copy()
    if window.empty:
        raise ValueError("cut_end is before the first bar")
    sentinel = window.index[0] - pd.Timedelta(minutes=1)
    prior = pd.DataFrame(
        {
            "open": [float(prior_close)],
            "high": [float(prior_close)],
            "low": [float(prior_close)],
            "close": [float(prior_close)],
            "volume": [0.0],
            "is_prior_close": [True],
        },
        index=pd.DatetimeIndex([sentinel]),
    )
    window["is_prior_close"] = False
    return pd.concat([prior, window])


def label_cut(
    bars: pd.DataFrame,
    cut_end: pd.Timestamp,
    entry_close: float,
    *,
    side: str | None = None,
    anchor_time: pd.Timestamp | None = None,
    sell_target: float | None = None,
    label_horizon_minutes: int | None = None,
    stop_loss: float | None = None,
    rise_pct: float | None = None,
) -> tuple[int, float | None]:
    """Ternary label for a cut relative to the valley dip.

    * ``-1`` if the cut is on the plunging / left side of the dip
      (``side=="before"``, or ``cut_end`` strictly before ``anchor_time``).
    * ``1`` if, walking bars after the cut within ``label_horizon_minutes``
      (default 15 / env), the fee-aware sell target is touched **before**
      the stop.
    * ``0`` if the stop is touched first, both print on the same bar, or the
      target is never reached in the horizon.

    ``rise_pct`` is accepted only for backward-compatible callers and is
    ignored. Returns ``(label, max high in the label horizon)``.
    """
    del rise_pct  # unused; kept so older call sites still type-check at runtime
    work = _sorted_bars(bars)
    end = _as_index_timestamp(cut_end, work.index) if not work.empty else pd.Timestamp(cut_end)

    plunging = False
    if side is not None:
        plunging = str(side).lower() == "before"
    elif anchor_time is not None and not work.empty:
        anchor = _as_index_timestamp(anchor_time, work.index)
        plunging = end < anchor
    elif anchor_time is not None:
        plunging = pd.Timestamp(cut_end) < pd.Timestamp(anchor_time)

    horizon = max(
        0,
        int(
            label_horizon_minutes
            if label_horizon_minutes is not None
            else get_label_horizon_minutes()
        ),
    )
    stop_frac = max(
        0.0,
        float(stop_loss if stop_loss is not None else get_label_stop_loss()),
    )

    window = pd.DataFrame()
    max_high: float | None = None
    if not work.empty:
        future = work.loc[work.index > end]
        horizon_end = end + pd.Timedelta(minutes=horizon)
        window = future.loc[future.index <= horizon_end]
        if not window.empty and "high" in window.columns:
            window_high = window["high"].astype(float)
            window_high = window_high[np.isfinite(window_high.to_numpy())]
            if not window_high.empty:
                max_high = float(window_high.max())

    if plunging:
        return -1, max_high

    if entry_close is None or not math.isfinite(entry_close) or entry_close <= 0:
        return 0, max_high

    if sell_target is None:
        target = target_sell_price_for_entry(float(entry_close))
    else:
        target = float(sell_target)
    if not math.isfinite(target):
        return 0, max_high

    stop_price = (
        float(entry_close) * (1.0 - stop_frac) if stop_frac > 0.0 else None
    )

    if window.empty:
        return 0, max_high

    for _, bar in window.iterrows():
        high = float(bar["high"]) if "high" in bar.index and pd.notna(bar["high"]) else math.nan
        low = float(bar["low"]) if "low" in bar.index and pd.notna(bar["low"]) else math.nan
        hit_target = math.isfinite(high) and high + 1e-9 >= target
        hit_stop = (
            stop_price is not None
            and math.isfinite(low)
            and low <= stop_price + 1e-9
        )
        if hit_stop and hit_target:
            # Same 1m bar: path unknown → treat as stop-first (not a clean win).
            return 0, max_high
        if hit_stop:
            return 0, max_high
        if hit_target:
            return 1, max_high
    return 0, max_high


def target_sell_price_for_entry(
    entry_close: float,
    *,
    buy_notional_usd: float | None = None,
    net_return: float | None = None,
) -> float:
    """Fee-aware sell level that nets ``net_return`` from ``entry_close``.

    Defaults come from ``get_buy_notional_usd()`` /
    ``get_target_return()`` (``.env`` /
    ``TRADINGAI_OPEN_DIP_BUY_NOTIONAL`` /
    ``TRADINGAI_OPEN_DIP_TARGET_RETURN``).
    """
    price = float(entry_close)
    if price <= 0.0 or not math.isfinite(price):
        raise ValueError("entry_close must be a positive finite price.")
    notional = (
        float(buy_notional_usd)
        if buy_notional_usd is not None
        else float(get_buy_notional_usd())
    )
    ret = float(net_return) if net_return is not None else float(get_target_return())
    result = fee_aware_sell_price(
        price,
        buy_notional_usd=notional,
        net_return=ret,
    )
    return float(result["sell_price"])


def count_level_crossings(bars: pd.DataFrame, level: float) -> int:
    """Count upward close-to-close crosses of ``level`` (below → above).

    Only transitions from a below-level close to an above-level close count.
    Downward crosses (above → below) are ignored. Exact touches
    (close == level) do not change side until price leaves to a non-zero side.
    """
    if bars is None or bars.empty or "close" not in bars.columns:
        return 0
    if level is None or not math.isfinite(level):
        return 0
    closes = _sorted_bars(bars)["close"].astype(float)
    closes = closes[np.isfinite(closes.to_numpy())]
    if len(closes) < 2:
        return 0
    signs = np.sign(closes.to_numpy() - float(level))
    crossings = 0
    prev = 0.0
    for sign in signs:
        if sign == 0.0:
            continue
        if prev != 0.0 and sign != prev and sign > 0.0:
            crossings += 1
        prev = float(sign)
    return int(crossings)


def recent_session_window(
    session_history: list[pd.DataFrame],
    cut_end: pd.Timestamp,
    *,
    max_sessions: int = DEFAULT_CROSSING_SESSIONS,
) -> pd.DataFrame:
    """Last ``max_sessions`` RTH frames, with the current session cut at ``cut_end``.

    Older sessions are included in full. The newest frame is truncated to bars
    at or before ``cut_end`` so the crossing count has no post-cut lookahead.
    """
    if not session_history:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    window = list(session_history[-max(1, int(max_sessions)) :])
    pieces: list[pd.DataFrame] = []
    for index, frame in enumerate(window):
        ordered = _sorted_bars(frame)
        if ordered.empty:
            continue
        if index == len(window) - 1:
            end = _as_index_timestamp(cut_end, ordered.index)
            ordered = ordered.loc[ordered.index <= end]
        if not ordered.empty:
            pieces.append(ordered[["open", "high", "low", "close", "volume"]])
    if not pieces:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pieces).sort_index()


def split_rth_sessions(bars: pd.DataFrame) -> list[tuple[date, pd.DataFrame]]:
    """Split a ticker history into regular-session frames keyed by ET date."""
    work = _sorted_bars(bars)
    if work.empty:
        return []
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("Intraday bars require a DatetimeIndex.")
    sessions: list[tuple[date, pd.DataFrame]] = []
    for day, group in work.groupby(work.index.date, sort=True):
        session = filter_rth_session(group, session_day=day)
        if not session.empty:
            sessions.append((day, session))
    return sessions


def assemble_cuts(
    sessions: list[tuple[str, date, pd.DataFrame, float]],
    *,
    dip_pct: float = DEFAULT_DATASET_DIP_PCT,
    rise_pct: float = DEFAULT_DATASET_RISE_PCT,
    n_cuts: int = DEFAULT_CUTS_PER_SESSION,
    cuts_before: int | None = None,
    cuts_after: int | None = None,
    min_horizon_minutes: int | None = None,
    label_horizon_minutes: int | None = None,
    stop_loss: float | None = None,
    target_return: float | None = None,
    buy_notional_usd: float | None = None,
    cut_min_minutes_from_open: int = DEFAULT_CUT_MIN_MINUTES_FROM_OPEN,
    cut_max_minutes_from_open: int = DEFAULT_CUT_MAX_MINUTES_FROM_OPEN,
    cut_vicinity_minutes: int | None = None,  # ignored; kept for call-site compat
    valley_window_minutes: int = DEFAULT_VALLEY_WINDOW_MINUTES,
    valley_order_minutes: int = DEFAULT_VALLEY_ORDER_MINUTES,
    crossing_sessions: int = DEFAULT_CROSSING_SESSIONS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build sample and bar frames. The third value is the qualifying session count.

    ``sessions`` items are ``(ticker, session_date, rth_bars, prior_close)``.
    One seeded generator is advanced in the given session order.

    The dip is the most pronounced valley in the first
    ``valley_window_minutes``. Cut ends are ``n_cuts`` random bars in
    ``[open + cut_min_minutes_from_open, open + cut_max_minutes_from_open]``
    (default 1.0–2.5h). After-dip cuts use ``label_horizon_minutes``
    (default 15): label ``1`` if the fee-aware sell target prints before the
    stop, else ``0``. Cut eligibility keeps at least
    ``max(min_horizon_minutes, label_horizon_minutes)`` bars after the cut.

    ``cuts_before`` / ``cuts_after`` (legacy) sum into ``n_cuts`` when both are
    set. ``cut_vicinity_minutes`` is ignored.

    Label defaults resolve from ``.env`` via ``get_label_*`` /
    ``get_target_return`` / ``get_buy_notional_usd`` when arguments are
    omitted. ``sell_crossings_5d`` counts upward close-to-close crosses
    (below → above) of the fee-aware target sell price over the last
    ``crossing_sessions`` RTH sessions through ``cut_end``.
    """
    resolved_horizon = int(
        label_horizon_minutes
        if label_horizon_minutes is not None
        else get_label_horizon_minutes()
    )
    resolved_min_horizon = int(
        min_horizon_minutes
        if min_horizon_minutes is not None
        else max(DEFAULT_MIN_HORIZON_MINUTES, resolved_horizon)
    )
    resolved_stop = float(
        stop_loss if stop_loss is not None else get_label_stop_loss()
    )
    resolved_return = float(
        target_return if target_return is not None else get_target_return()
    )
    resolved_notional = float(
        buy_notional_usd
        if buy_notional_usd is not None
        else get_buy_notional_usd()
    )
    if cuts_before is not None or cuts_after is not None:
        resolved_cuts = int(cuts_before or 0) + int(cuts_after or 0)
    else:
        resolved_cuts = int(n_cuts)
    _ = cut_vicinity_minutes  # legacy no-op
    rng = np.random.default_rng(seed)
    sample_rows: list[dict] = []
    bar_rows: list[dict] = []
    qualifying = 0
    sample_id = 0
    horizon_for_cuts = max(resolved_min_horizon, resolved_horizon)
    session_histories: dict[str, deque[pd.DataFrame]] = defaultdict(
        lambda: deque(maxlen=max(1, int(crossing_sessions)))
    )
    for ticker, session_day, bars, prior_close in sessions:
        key = ticker.upper()
        session_histories[key].append(bars)
        anchor = find_dip_anchor(
            bars,
            prior_close,
            dip_pct,
            valley_window_minutes=valley_window_minutes,
            valley_order_minutes=valley_order_minutes,
        )
        if anchor is None:
            continue
        cuts = sample_cut_ends(
            bars,
            anchor.anchor_time,
            resolved_cuts,
            horizon_for_cuts,
            rng,
            cut_min_minutes_from_open=cut_min_minutes_from_open,
            cut_max_minutes_from_open=cut_max_minutes_from_open,
        )
        qualifying += 1
        if not cuts:
            continue
        ordered = _sorted_bars(bars)
        history = list(session_histories[key])
        for cut_end, side in cuts:
            entry_close = float(ordered.loc[cut_end, "close"])
            sell_price = target_sell_price_for_entry(
                entry_close,
                buy_notional_usd=resolved_notional,
                net_return=resolved_return,
            )
            label, max_high = label_cut(
                ordered,
                cut_end,
                entry_close,
                side=side,
                anchor_time=anchor.anchor_time,
                sell_target=sell_price,
                label_horizon_minutes=resolved_horizon,
                stop_loss=resolved_stop,
            )
            crossing_bars = recent_session_window(
                history,
                cut_end,
                max_sessions=crossing_sessions,
            )
            crossings = count_level_crossings(crossing_bars, sell_price)
            stop_price = (
                entry_close * (1.0 - resolved_stop) if resolved_stop > 0.0 else None
            )
            sample_id += 1
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "ticker": key,
                    "session_date": session_day,
                    "cut_end": cut_end,
                    "side": side,
                    "prior_close": round(float(prior_close), 6),
                    "entry_close": round(entry_close, 6),
                    "anchor_time": anchor.anchor_time,
                    "dip_pct": anchor.dip_pct,
                    "max_future_high": None if max_high is None else round(max_high, 6),
                    "rise_pct": float(rise_pct),
                    "label": int(label),
                    "target_sell_price": round(sell_price, 6),
                    "stop_loss_price": (
                        None if stop_price is None else round(float(stop_price), 6)
                    ),
                    "sell_crossings_5d": int(crossings),
                }
            )
            cut = build_cut(ordered, prior_close, cut_end)
            for seq, (bar_time, row) in enumerate(cut.iterrows()):
                bar_rows.append(
                    {
                        "sample_id": sample_id,
                        "seq": seq,
                        "bar_time": pd.Timestamp(bar_time),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "is_prior_close": bool(row["is_prior_close"]),
                    }
                )
    samples = pd.DataFrame(sample_rows, columns=list(SAMPLE_COLUMNS))
    bar_frame = pd.DataFrame(bar_rows, columns=list(BAR_COLUMNS))
    return samples, bar_frame, qualifying


def write_dip_dataset(
    samples: pd.DataFrame,
    bars: pd.DataFrame,
    out_dir: Path | str,
) -> Path:
    """Write ``samples.parquet`` and ``bars.parquet`` under ``out_dir``."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sample_frame = samples if not samples.empty else pd.DataFrame(columns=list(SAMPLE_COLUMNS))
    bar_frame = bars if not bars.empty else pd.DataFrame(columns=list(BAR_COLUMNS))
    sample_frame.to_parquet(destination / "samples.parquet", index=False)
    bar_frame.to_parquet(destination / "bars.parquet", index=False)
    return destination
