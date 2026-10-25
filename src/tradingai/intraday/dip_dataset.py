"""Training cuts for a 4% intraday dip near a fee-aware sell-target recovery.

Every sample comes from a regular session that eventually trades at least
``dip_pct`` below the prior close. Cuts end at random 1-minute bars strictly
before and after the first bar that prints that dip, and only within
``cut_vicinity_minutes`` of that dip (default 45). The label uses only bars
after the cut:

* ``label``: ``1`` if the fee-aware sell target is reached (any high) within
  ``label_horizon_minutes`` after the cut (default 30); ``-1`` if a later
  session low prints a new trough (still falling); ``0`` neither.
* ``sell_crossings_5d``: how many times 1m closes crossed *up through* the
  fee-aware target sell price (from ``entry_close`` / ``DEFAULT_TARGET_RETURN``)
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
from tradingai.intraday.profit_target import (
    DEFAULT_BUY_NOTIONAL_USD,
    DEFAULT_TARGET_RETURN,
    fee_aware_sell_price,
)

DEFAULT_DATASET_DIP_PCT = 4.0
DEFAULT_DATASET_RISE_PCT = 0.4  # retained in sample rows for compatibility
DEFAULT_CUTS_BEFORE = 4
DEFAULT_CUTS_AFTER = 4
DEFAULT_LABEL_HORIZON_MINUTES = 30
DEFAULT_MIN_HORIZON_MINUTES = DEFAULT_LABEL_HORIZON_MINUTES
DEFAULT_CUT_VICINITY_MINUTES = 45
DEFAULT_CROSSING_SESSIONS = 5

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


def find_dip_anchor(
    bars: pd.DataFrame,
    prior_close: float,
    dip_pct: float = DEFAULT_DATASET_DIP_PCT,
) -> DipAnchor | None:
    """First bar whose running session low is at least ``dip_pct`` under prior close.

    ``dip_pct`` is a positive percent (4 means −4%). The returned ``dip_pct``
    is the running depth at that bar, in percent, and is negative.
    """
    work = _sorted_bars(bars)
    if work.empty or "low" not in work.columns:
        return None
    if prior_close is None or not math.isfinite(prior_close) or prior_close <= 0:
        return None
    running = work["low"].astype(float).cummin()
    depth_pct = (running / float(prior_close) - 1.0) * 100.0
    threshold = -abs(float(dip_pct))
    hit = depth_pct <= threshold + 1e-6
    if not bool(hit.any()):
        return None
    position = int(np.flatnonzero(hit.to_numpy())[0])
    anchor_time = pd.Timestamp(work.index[position])
    anchor_low = float(running.iloc[position])
    depth = round(float(depth_pct.iloc[position]), 4)
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
    n_before: int,
    n_after: int,
    min_horizon_minutes: int,
    rng: np.random.Generator,
    *,
    vicinity_minutes: int = DEFAULT_CUT_VICINITY_MINUTES,
) -> list[tuple[pd.Timestamp, str]]:
    """Random cut ends strictly before and after ``anchor_time``.

    A bar is eligible only when:
    * it is within ``vicinity_minutes`` of the dip anchor (exclusive of the
      anchor itself), and
    * at least ``min_horizon_minutes`` later bars remain.

    If a side has fewer eligible bars than requested, every eligible bar on
    that side is used. ``vicinity_minutes <= 0`` disables the window and keeps
    the whole-session pool (still excluding the anchor / horizon).
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    work = _sorted_bars(bars)
    if work.empty:
        return []
    horizon = max(0, int(min_horizon_minutes))
    last_eligible = len(work) - horizon - 1
    if last_eligible < 0:
        return []
    anchor = _as_index_timestamp(anchor_time, work.index)
    vicinity = max(0, int(vicinity_minutes))
    window = pd.Timedelta(minutes=vicinity) if vicinity > 0 else None
    before: list[int] = []
    after: list[int] = []
    for position in range(last_eligible + 1):
        stamp = work.index[position]
        if window is not None:
            delta = abs(pd.Timestamp(stamp) - anchor)
            if delta > window:
                continue
        if stamp < anchor:
            before.append(position)
        elif stamp > anchor:
            after.append(position)
    chosen: list[tuple[pd.Timestamp, str]] = []
    for position in _take_positions(before, n_before, rng):
        chosen.append((pd.Timestamp(work.index[position]), "before"))
    for position in _take_positions(after, n_after, rng):
        chosen.append((pd.Timestamp(work.index[position]), "after"))
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
    sell_target: float | None = None,
    label_horizon_minutes: int = DEFAULT_LABEL_HORIZON_MINUTES,
    rise_pct: float | None = None,
) -> tuple[int, float | None]:
    """Ternary label from post-cut bars only.

    * ``1`` if any high within ``label_horizon_minutes`` after ``cut_end``
      reaches the fee-aware ``sell_target`` (computed from ``entry_close`` when
      omitted).
    * ``-1`` if any later low (rest of session) is strictly below the running
      session low through ``cut_end`` and the sell target was not hit in time.
    * ``0`` otherwise.

    ``rise_pct`` is accepted only for backward-compatible callers and is
    ignored. Returns ``(label, max high in the label horizon)``.
    """
    del rise_pct  # unused; kept so older call sites still type-check at runtime
    if entry_close is None or not math.isfinite(entry_close) or entry_close <= 0:
        return 0, None
    work = _sorted_bars(bars)
    end = _as_index_timestamp(cut_end, work.index)
    through_cut = work.loc[work.index <= end]
    if through_cut.empty or "low" not in work.columns:
        return 0, None
    trough_so_far = float(through_cut["low"].astype(float).min())
    future = work.loc[work.index > end]
    if future.empty:
        return 0, None

    horizon = max(0, int(label_horizon_minutes))
    horizon_end = end + pd.Timedelta(minutes=horizon)
    window = future.loc[future.index <= horizon_end]
    window_high = window["high"].astype(float) if not window.empty else pd.Series(dtype=float)
    window_high = window_high[np.isfinite(window_high.to_numpy())]
    max_high = float(window_high.max()) if not window_high.empty else None

    if sell_target is None:
        target = target_sell_price_for_entry(float(entry_close))
    else:
        target = float(sell_target)
    if (
        max_high is not None
        and math.isfinite(target)
        and max_high + 1e-9 >= target
    ):
        return 1, max_high

    future_low = future["low"].astype(float)
    future_low = future_low[np.isfinite(future_low.to_numpy())]
    if not future_low.empty and float(future_low.min()) < trough_so_far - 1e-9:
        return -1, max_high
    return 0, max_high


def target_sell_price_for_entry(
    entry_close: float,
    *,
    buy_notional_usd: float = DEFAULT_BUY_NOTIONAL_USD,
    net_return: float = DEFAULT_TARGET_RETURN,
) -> float:
    """Fee-aware sell level that nets ``net_return`` from ``entry_close``."""
    price = float(entry_close)
    if price <= 0.0 or not math.isfinite(price):
        raise ValueError("entry_close must be a positive finite price.")
    result = fee_aware_sell_price(
        price,
        buy_notional_usd=float(buy_notional_usd),
        net_return=float(net_return),
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
    cuts_before: int = DEFAULT_CUTS_BEFORE,
    cuts_after: int = DEFAULT_CUTS_AFTER,
    min_horizon_minutes: int = DEFAULT_MIN_HORIZON_MINUTES,
    label_horizon_minutes: int = DEFAULT_LABEL_HORIZON_MINUTES,
    cut_vicinity_minutes: int = DEFAULT_CUT_VICINITY_MINUTES,
    crossing_sessions: int = DEFAULT_CROSSING_SESSIONS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build sample and bar frames. The third value is the qualifying session count.

    ``sessions`` items are ``(ticker, session_date, rth_bars, prior_close)``.
    One seeded generator is advanced in the given session order.

    Cut ends are drawn within ``cut_vicinity_minutes`` of the dip anchor.
    Positive labels require the fee-aware sell target within
    ``label_horizon_minutes`` after the cut. Cut eligibility keeps at least
    ``max(min_horizon_minutes, label_horizon_minutes)`` bars after the cut.

    ``sell_crossings_5d`` counts upward close-to-close crosses (below → above)
    of the fee-aware target sell price (from ``entry_close`` /
    ``DEFAULT_TARGET_RETURN``) over the last ``crossing_sessions`` RTH sessions
    through ``cut_end``. Downward crosses do not count.
    """
    rng = np.random.default_rng(seed)
    sample_rows: list[dict] = []
    bar_rows: list[dict] = []
    qualifying = 0
    sample_id = 0
    horizon_for_cuts = max(int(min_horizon_minutes), int(label_horizon_minutes))
    session_histories: dict[str, deque[pd.DataFrame]] = defaultdict(
        lambda: deque(maxlen=max(1, int(crossing_sessions)))
    )
    for ticker, session_day, bars, prior_close in sessions:
        key = ticker.upper()
        session_histories[key].append(bars)
        anchor = find_dip_anchor(bars, prior_close, dip_pct)
        if anchor is None:
            continue
        cuts = sample_cut_ends(
            bars,
            anchor.anchor_time,
            cuts_before,
            cuts_after,
            horizon_for_cuts,
            rng,
            vicinity_minutes=cut_vicinity_minutes,
        )
        qualifying += 1
        if not cuts:
            continue
        ordered = _sorted_bars(bars)
        history = list(session_histories[key])
        for cut_end, side in cuts:
            entry_close = float(ordered.loc[cut_end, "close"])
            sell_price = target_sell_price_for_entry(entry_close)
            label, max_high = label_cut(
                ordered,
                cut_end,
                entry_close,
                sell_target=sell_price,
                label_horizon_minutes=label_horizon_minutes,
            )
            crossing_bars = recent_session_window(
                history,
                cut_end,
                max_sessions=crossing_sessions,
            )
            crossings = count_level_crossings(crossing_bars, sell_price)
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
