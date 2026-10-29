"""Sample charts for dip-recovery training cuts.

Each panel shows the full RTH session with the training cut highlighted up to
``cut_end``. Post-cut bars stay visible so the label context is readable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_NAME = {-1: "falling", 0: "neither", 1: "recover"}
LABEL_COLOR = {-1: "#c0392b", 0: "#7f8c8d", 1: "#27ae60"}

SessionKey = tuple[str, date]
SessionMap = Mapping[SessionKey, pd.DataFrame]
SessionLoader = Callable[[str, date], pd.DataFrame | None]


def _session_day(value) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def _as_series_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize session OHLCV to a DatetimeIndex sorted ascending."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if isinstance(frame.index, pd.DatetimeIndex):
        work = frame.sort_index()
    elif "bar_time" in frame.columns:
        work = frame.set_index(pd.to_datetime(frame["bar_time"])).sort_index()
    else:
        raise ValueError("Session bars need a DatetimeIndex or bar_time column")
    if work.index.has_duplicates:
        work = work[~work.index.duplicated(keep="last")]
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in work.columns]
    return work[cols].astype(float)


def _align_cut_end(cut_end, index: pd.DatetimeIndex) -> pd.Timestamp:
    stamp = pd.Timestamp(cut_end)
    if index.tz is None:
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("America/New_York").tz_localize(None)
        return stamp
    if stamp.tzinfo is None:
        return stamp.tz_localize(index.tz)
    return stamp.tz_convert(index.tz)


def pick_stratified_samples(
    samples: pd.DataFrame,
    *,
    n: int = 20,
    seed: int = 0,
    per_label: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Prefer unique tickers within each label bucket."""
    if samples is None or samples.empty or n <= 0:
        return pd.DataFrame(columns=samples.columns if samples is not None else [])
    quotas = dict(per_label) if per_label is not None else {-1: 8, 1: 8, 0: 4}
    # Scale quotas if caller asks for a different total.
    quota_sum = sum(max(0, int(v)) for v in quotas.values())
    if quota_sum != n and quota_sum > 0:
        scaled = {lab: max(0, int(round(n * q / quota_sum))) for lab, q in quotas.items()}
        # Fix rounding drift on the largest bucket.
        drift = n - sum(scaled.values())
        if drift != 0:
            top = max(scaled, key=scaled.get)
            scaled[top] = max(0, scaled[top] + drift)
        quotas = scaled
    rng = np.random.default_rng(seed)
    picked: list[pd.Series] = []
    for lab, count in quotas.items():
        if count <= 0:
            continue
        pool = samples[samples["label"] == lab]
        if pool.empty:
            continue
        tickers = list(pool["ticker"].astype(str).unique())
        rng.shuffle(tickers)
        take: list[pd.Series] = []
        for ticker in tickers:
            row = pool[pool["ticker"] == ticker].sample(
                1, random_state=int(rng.integers(0, 1_000_000))
            )
            take.append(row.iloc[0])
            if len(take) >= count:
                break
        if len(take) < count:
            used = {int(r["sample_id"]) for r in take}
            extra = pool[~pool["sample_id"].isin(used)]
            need = count - len(take)
            if not extra.empty:
                for _, row in extra.sample(
                    min(need, len(extra)), random_state=int(rng.integers(0, 1_000_000))
                ).iterrows():
                    take.append(row)
        picked.extend(take[:count])
    if not picked:
        return samples.sample(min(n, len(samples)), random_state=seed).reset_index(drop=True)
    return pd.DataFrame(picked).reset_index(drop=True)


def resolve_session_bars(
    ticker: str,
    session_date,
    *,
    sessions: SessionMap | None = None,
    session_loader: SessionLoader | None = None,
) -> pd.DataFrame:
    day = _session_day(session_date)
    key = (str(ticker).upper(), day)
    if sessions is not None and key in sessions:
        return _as_series_index(sessions[key])
    if session_loader is not None:
        loaded = session_loader(str(ticker).upper(), day)
        if loaded is not None and not loaded.empty:
            return _as_series_index(loaded)
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def plot_dip_sample(
    ax,
    sample: pd.Series | Mapping,
    *,
    session_bars: pd.DataFrame,
    cut_bars: pd.DataFrame | None = None,
    show_legend: bool = True,
    title_fontsize: float = 9,
) -> None:
    """Draw one sample: full session closes, cut window emphasized, markers."""
    row = sample if isinstance(sample, pd.Series) else pd.Series(sample)
    label = int(row["label"])
    color = LABEL_COLOR.get(label, "#7f8c8d")
    session = _as_series_index(session_bars)

    cut_end = None
    if session.empty and cut_bars is not None and not cut_bars.empty:
        # Fallback: cut-only view when session history is unavailable.
        real = cut_bars[~cut_bars["is_prior_close"].astype(bool)].sort_values("seq")
        times = pd.to_datetime(real["bar_time"])
        closes = real["close"].astype(float)
        ax.fill_between(
            times,
            real["low"].astype(float),
            real["high"].astype(float),
            color="#bdc3c7",
            alpha=0.35,
            label="H-L (cut)",
        )
        ax.plot(times, closes, color="#2c3e50", lw=1.4, label="close (cut)")
        if len(times):
            cut_end = times.iloc[-1]
            ax.scatter([cut_end], [closes.iloc[-1]], c=color, s=48, zorder=5, label="cut end")
    else:
        times = session.index
        closes = session["close"]
        cut_end = _align_cut_end(row["cut_end"], times)
        pre = times <= cut_end
        post = times > cut_end

        if "low" in session.columns and "high" in session.columns:
            ax.fill_between(
                times,
                session["low"],
                session["high"],
                color="#d5d8dc",
                alpha=0.35,
                label="H-L (session)",
            )
        if bool(pre.any()):
            ax.plot(
                times[pre],
                closes[pre],
                color="#2c3e50",
                lw=1.5,
                label="close (cut)",
                zorder=3,
            )
        if bool(post.any()):
            ax.plot(
                times[post],
                closes[post],
                color="#95a5a6",
                lw=1.2,
                label="close (after cut)",
                zorder=2,
            )
        # Soft vertical divider so the cut boundary is obvious on a full day.
        ax.axvline(cut_end, color=color, ls="--", lw=1.0, alpha=0.85, label="cut end")
        if cut_end in closes.index:
            ax.scatter(
                [cut_end],
                [float(closes.loc[cut_end])],
                c=color,
                s=48,
                zorder=5,
            )
        elif bool(pre.any()):
            ax.scatter(
                [times[pre][-1]],
                [float(closes[pre].iloc[-1])],
                c=color,
                s=48,
                zorder=5,
            )

    prior = float(row["prior_close"])
    sell = float(row["target_sell_price"])
    ax.axhline(prior, color="#2980b9", ls="--", lw=1.1, label=f"prior {prior:.2f}")
    ax.axhline(sell, color="#e67e22", ls=":", lw=1.3, label=f"sell tgt {sell:.2f}")

    if "anchor_time" in row and pd.notna(row["anchor_time"]) and not session.empty:
        anchor = _align_cut_end(row["anchor_time"], session.index)
        if session.index[0] <= anchor <= session.index[-1]:
            ax.axvline(anchor, color="#8e44ad", ls=":", lw=0.9, alpha=0.7, label="dip anchor")

    ax.set_title(
        f"{row['ticker']} {row['session_date']} side={row['side']}  "
        f"label={label} ({LABEL_NAME.get(label, '?')})  "
        f"sell_crossings_up_5d={int(row['sell_crossings_5d'])}  "
        f"dip={float(row['dip_pct']):.2f}%",
        fontsize=title_fontsize,
        color=color,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("price")
    if show_legend:
        ax.legend(fontsize=7, loc="best")


def write_dip_dataset_viz(
    samples: pd.DataFrame,
    cut_bars: pd.DataFrame,
    out_dir: Path | str,
    *,
    sessions: SessionMap | None = None,
    session_loader: SessionLoader | None = None,
    n: int = 20,
    seed: int = 0,
) -> Path:
    """Write overview grid + per-sample PNGs under ``out_dir``.

    Uses the Agg backend and avoids ``constrained_layout`` (it can hang for
    minutes on multi-panel date axes with full-session fills).
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.glob("sample_*.png"):
        path.unlink()

    selected = pick_stratified_samples(samples, n=n, seed=seed)
    if selected.empty:
        return destination

    selected.to_csv(destination / f"selected_{len(selected)}.csv", index=False)
    print(f"dip viz: plotting {len(selected)} samples → {destination}", flush=True)

    # Index once; full cut_bars frames are large after a universe build.
    bars_by_id = {
        int(sid): frame.sort_values("seq")
        for sid, frame in cut_bars.groupby("sample_id", sort=False)
        if int(sid) in set(int(x) for x in selected["sample_id"])
    }

    def _session_for(sample: pd.Series) -> pd.DataFrame:
        return resolve_session_bars(
            str(sample["ticker"]),
            sample["session_date"],
            sessions=sessions,
            session_loader=session_loader,
        )

    # Per-sample panels first (simple single-axis figures; fail soft).
    for i, sample in selected.iterrows():
        sid = int(sample["sample_id"])
        fig, ax = plt.subplots(figsize=(8, 3.4))
        try:
            plot_dip_sample(
                ax,
                sample,
                session_bars=_session_for(sample),
                cut_bars=bars_by_id.get(sid),
            )
            fig.tight_layout()
            fig.savefig(
                destination / f"sample_{i:02d}_{sample['ticker']}_{int(sample['label'])}.png",
                dpi=120,
            )
        finally:
            plt.close(fig)
        print(f"dip viz: sample {i + 1}/{len(selected)}", flush=True)

    # Overview grid without constrained_layout (avoids matplotlib hang).
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
    flat = np.atleast_1d(axes).ravel()
    try:
        for i, sample in selected.iterrows():
            ax = flat[i]
            sid = int(sample["sample_id"])
            plot_dip_sample(
                ax,
                sample,
                session_bars=_session_for(sample),
                cut_bars=bars_by_id.get(sid),
                show_legend=(i == 0),
                title_fontsize=8,
            )
            ax.tick_params(labelsize=7)
        for j in range(len(selected), len(flat)):
            flat[j].axis("off")
        label_counts = samples["label"].value_counts()
        fig.suptitle(
            f"Dip training cuts ({len(selected)} of {len(samples)}; full session; "
            f"labels −1={int(label_counts.get(-1, 0))}, "
            f"0={int(label_counts.get(0, 0))}, 1={int(label_counts.get(1, 0))})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(destination / f"overview_{len(selected)}.png", dpi=120)
        print(f"dip viz: wrote overview_{len(selected)}.png", flush=True)
    finally:
        plt.close(fig)

    return destination
