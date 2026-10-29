"""Simulator config: loaded from ``dip_sim.toml`` (not from ``.env``)."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SIM_CONFIG_PATH = _PACKAGE_DIR / "dip_sim.toml"
_SCHEMA_PATH = _PACKAGE_DIR / "schema.sql"
_DATA_UNIVERSE_DIR = _PACKAGE_DIR.parent / "data" / "universe"
DEFAULT_SIM_UNIVERSE_FILE = _DATA_UNIVERSE_DIR / "us_equities_3000.txt"

# Fallback constants if the TOML file is missing (keep in sync with dip_sim.toml).
DEFAULT_SIM_DATABASE_URL = "postgresql://localhost:5432/tradingai_sim"
DEFAULT_DOWNLOAD_DAYS = 60
DEFAULT_DOWNLOAD_PACE_SECONDS = 0.25
DEFAULT_STOP_LOSS = 0.01
DEFAULT_REQUIRE_STATUS = ("reclaiming",)
DEFAULT_MAX_POSITIONS = 5
DEFAULT_ENTRY_CUTOFF_HOUR = 11
DEFAULT_ENTRY_CUTOFF_MINUTE = 0
DEFAULT_START_CASH = 100_000.0
DEFAULT_MIN_BOUNCE_SCORE: float | None = None
DEFAULT_EXCLUDE_FIRST_BAR_GREEN = False
DEFAULT_USE_TRAIL = True
DEFAULT_DIP_PCT = 2.0
DEFAULT_DIP_WINDOW_MINUTES = 30
DEFAULT_BUY_NOTIONAL_USD = 20_000.0
DEFAULT_TARGET_RETURN = (1.0 / 365.0) * (7.0 / 5.0)
DEFAULT_PROFIT_TAKER_BUFFER = 0.0002
# IBKR-like hist query virtual timing (60 req / 10 min → ~10s pace).
DEFAULT_QUERY_LATENCY_SECONDS = 0.5
DEFAULT_QUERY_PACE_SECONDS = 10.0
DEFAULT_SCAN_BATCH_SIZE = 20
# Bounce/reclaim momentum lookback for the sim only (live scan keeps 15m).
DEFAULT_SIM_MOMENTUM_MINUTES = 5
# Max RMSE of asymmetric parabola fit / dip depth on the as-of cut.
DEFAULT_MAX_PARABOLA_REL_RMSE = 0.15
DEFAULT_EXPORT_CHARTS = True
DEFAULT_CHARTS_DIR = "data/sim_charts"


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


@lru_cache
def get_sim_config_path() -> Path:
    return DEFAULT_SIM_CONFIG_PATH


def clear_sim_config_cache() -> None:
    """Drop cached config (tests / after swapping --config)."""
    get_sim_config_path.cache_clear()
    load_sim_config.cache_clear()


@lru_cache
def load_sim_config(path: str | None = None) -> dict[str, Any]:
    """Load sim parameters from TOML. ``path`` overrides the packaged default."""
    cfg_path = Path(path).expanduser() if path else get_sim_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Sim config not found: {cfg_path}. "
            "Expected packaged dip_sim.toml or pass --config."
        )
    return _read_toml(cfg_path)


def _section(name: str, path: str | None = None) -> dict[str, Any]:
    raw = load_sim_config(path)
    section = raw.get(name)
    return section if isinstance(section, dict) else {}


def get_sim_database_url(*, config_path: str | None = None) -> str:
    url = _section("database", config_path).get("url")
    if url:
        return str(url).strip()
    return DEFAULT_SIM_DATABASE_URL


def get_download_days(*, config_path: str | None = None) -> int:
    raw = _section("download", config_path).get("days")
    if raw is None:
        return DEFAULT_DOWNLOAD_DAYS
    return max(1, int(raw))


def get_download_pace_seconds(*, config_path: str | None = None) -> float:
    raw = _section("download", config_path).get("pace_seconds")
    if raw is None:
        return DEFAULT_DOWNLOAD_PACE_SECONDS
    return max(0.0, float(raw))


def resolve_sim_universe_path(*, config_path: str | None = None) -> Path:
    """Resolve ``[download].universe_file`` to an on-disk ticker list path."""
    raw = _section("download", config_path).get("universe_file")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_SIM_UNIVERSE_FILE
    path = Path(str(raw).strip()).expanduser()
    if path.is_file():
        return path
    if not path.is_absolute():
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.is_file():
            return cwd_candidate
        packaged = _DATA_UNIVERSE_DIR / path
        if packaged.is_file():
            return packaged
        by_name = _DATA_UNIVERSE_DIR / path.name
        if by_name.is_file():
            return by_name
    return path


def get_sim_download_universe(*, config_path: str | None = None) -> tuple[str, ...]:
    """Equity universe for ``tradingai-sim-download`` (not the live sync list)."""
    from tradingai.data.config import load_tickers_from_file

    path = resolve_sim_universe_path(config_path=config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Sim download universe not found: {path}. "
            "Set [download].universe_file in dip_sim.toml."
        )
    tickers = load_tickers_from_file(path)
    if not tickers:
        raise ValueError(f"Sim download universe is empty: {path}")
    return tickers


def get_sim_stop_loss(*, config_path: str | None = None) -> float:
    raw = _section("strategy", config_path).get("stop_loss")
    if raw is None:
        return DEFAULT_STOP_LOSS
    return max(0.0, float(raw))


def get_sim_dip_pct(*, config_path: str | None = None) -> float:
    raw = _section("strategy", config_path).get("dip_pct")
    if raw is None:
        return DEFAULT_DIP_PCT
    return max(0.1, float(raw))


def get_sim_window_minutes(*, config_path: str | None = None) -> int:
    raw = _section("strategy", config_path).get("window_minutes")
    if raw is None:
        return DEFAULT_DIP_WINDOW_MINUTES
    return max(5, int(raw))


def get_sim_target_return(*, config_path: str | None = None) -> float:
    raw = _section("strategy", config_path).get("target_return")
    if raw is None:
        return DEFAULT_TARGET_RETURN
    return max(0.0, float(raw))


def get_sim_buy_notional(*, config_path: str | None = None) -> float:
    raw = _section("strategy", config_path).get("buy_notional_usd")
    if raw is None:
        return DEFAULT_BUY_NOTIONAL_USD
    return max(1.0, float(raw))


def get_sim_max_positions(*, config_path: str | None = None) -> int:
    raw = _section("strategy", config_path).get("max_positions")
    if raw is None:
        return DEFAULT_MAX_POSITIONS
    return max(1, int(raw))


def get_sim_start_cash(*, config_path: str | None = None) -> float:
    raw = _section("strategy", config_path).get("start_cash")
    if raw is None:
        return DEFAULT_START_CASH
    return max(1.0, float(raw))


def get_sim_profit_taker_buffer(*, config_path: str | None = None) -> float:
    raw = _section("strategy", config_path).get("profit_taker_buffer")
    if raw is None:
        return DEFAULT_PROFIT_TAKER_BUFFER
    return max(0.0, float(raw))


def get_sim_require_status(*, config_path: str | None = None) -> tuple[str, ...]:
    raw = _section("strategy", config_path).get("require_status")
    if not raw:
        return DEFAULT_REQUIRE_STATUS
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in raw if str(p).strip()]
    return tuple(parts) if parts else DEFAULT_REQUIRE_STATUS


def get_sim_exclude_first_bar_green(*, config_path: str | None = None) -> bool:
    raw = _section("strategy", config_path).get("exclude_first_bar_green")
    if raw is None:
        return DEFAULT_EXCLUDE_FIRST_BAR_GREEN
    return bool(raw)


def get_sim_use_trail(*, config_path: str | None = None) -> bool:
    raw = _section("strategy", config_path).get("use_trail")
    if raw is None:
        return DEFAULT_USE_TRAIL
    return bool(raw)


def get_sim_min_bounce_score(*, config_path: str | None = None) -> float | None:
    section = _section("strategy", config_path)
    if "min_bounce_score" not in section:
        return DEFAULT_MIN_BOUNCE_SCORE
    raw = section.get("min_bounce_score")
    if raw is None or raw == "":
        return None
    return float(raw)


def get_sim_sell_trail(*, config_path: str | None = None) -> float | None:
    section = _section("strategy", config_path)
    if "sell_trail" not in section:
        return None
    raw = section.get("sell_trail")
    if raw is None or raw == "":
        return None
    return max(0.0, float(raw))


def get_sim_entry_cutoff(
    *, config_path: str | None = None
) -> tuple[int, int]:
    section = _section("strategy", config_path)
    hour = section.get("entry_cutoff_hour", DEFAULT_ENTRY_CUTOFF_HOUR)
    minute = section.get("entry_cutoff_minute", DEFAULT_ENTRY_CUTOFF_MINUTE)
    return int(hour), int(minute)


def get_sim_momentum_minutes(*, config_path: str | None = None) -> int:
    """Momentum lookback minutes for sim bounce/reclaim scoring."""
    raw = _section("strategy", config_path).get("momentum_minutes")
    if raw is None:
        return DEFAULT_SIM_MOMENTUM_MINUTES
    return max(1, int(raw))


def get_sim_max_parabola_rel_rmse(*, config_path: str | None = None) -> float:
    """Max relative RMSE for the asymmetric parabola dip-shape fit."""
    raw = _section("strategy", config_path).get("max_parabola_rel_rmse")
    if raw is None:
        return DEFAULT_MAX_PARABOLA_REL_RMSE
    return max(0.0, float(raw))


def get_query_latency_seconds(*, config_path: str | None = None) -> float:
    raw = _section("ibkr_query", config_path).get("query_latency_seconds")
    if raw is None:
        return DEFAULT_QUERY_LATENCY_SECONDS
    return max(0.0, float(raw))


def get_query_pace_seconds(*, config_path: str | None = None) -> float:
    raw = _section("ibkr_query", config_path).get("query_pace_seconds")
    if raw is None:
        return DEFAULT_QUERY_PACE_SECONDS
    return max(0.0, float(raw))


def get_scan_batch_size(*, config_path: str | None = None) -> int:
    raw = _section("ibkr_query", config_path).get("scan_batch_size")
    if raw is None:
        return DEFAULT_SCAN_BATCH_SIZE
    return max(1, int(raw))


def get_export_charts(*, config_path: str | None = None) -> bool:
    section = _section("charts", config_path)
    if "enabled" not in section:
        return DEFAULT_EXPORT_CHARTS
    return bool(section.get("enabled"))


def get_charts_dir(*, config_path: str | None = None) -> str:
    raw = _section("charts", config_path).get("dir")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_CHARTS_DIR
    return str(raw).strip()


def get_schema_sql() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8")


__all__ = [
    "DEFAULT_BUY_NOTIONAL_USD",
    "DEFAULT_CHARTS_DIR",
    "DEFAULT_DIP_PCT",
    "DEFAULT_DIP_WINDOW_MINUTES",
    "DEFAULT_DOWNLOAD_DAYS",
    "DEFAULT_DOWNLOAD_PACE_SECONDS",
    "DEFAULT_ENTRY_CUTOFF_HOUR",
    "DEFAULT_ENTRY_CUTOFF_MINUTE",
    "DEFAULT_EXCLUDE_FIRST_BAR_GREEN",
    "DEFAULT_EXPORT_CHARTS",
    "DEFAULT_MAX_PARABOLA_REL_RMSE",
    "DEFAULT_MAX_POSITIONS",
    "DEFAULT_SIM_MOMENTUM_MINUTES",
    "DEFAULT_MIN_BOUNCE_SCORE",
    "DEFAULT_PROFIT_TAKER_BUFFER",
    "DEFAULT_QUERY_LATENCY_SECONDS",
    "DEFAULT_QUERY_PACE_SECONDS",
    "DEFAULT_SCAN_BATCH_SIZE",
    "DEFAULT_REQUIRE_STATUS",
    "DEFAULT_SIM_CONFIG_PATH",
    "DEFAULT_SIM_DATABASE_URL",
    "DEFAULT_SIM_UNIVERSE_FILE",
    "DEFAULT_START_CASH",
    "DEFAULT_STOP_LOSS",
    "DEFAULT_TARGET_RETURN",
    "DEFAULT_USE_TRAIL",
    "clear_sim_config_cache",
    "get_charts_dir",
    "get_download_days",
    "get_download_pace_seconds",
    "get_export_charts",
    "get_query_latency_seconds",
    "get_query_pace_seconds",
    "get_scan_batch_size",
    "get_schema_sql",
    "get_sim_buy_notional",
    "get_sim_config_path",
    "get_sim_database_url",
    "get_sim_dip_pct",
    "get_sim_download_universe",
    "get_sim_entry_cutoff",
    "get_sim_exclude_first_bar_green",
    "get_sim_max_parabola_rel_rmse",
    "get_sim_max_positions",
    "get_sim_momentum_minutes",
    "get_sim_min_bounce_score",
    "get_sim_profit_taker_buffer",
    "get_sim_require_status",
    "get_sim_sell_trail",
    "get_sim_start_cash",
    "get_sim_stop_loss",
    "get_sim_target_return",
    "get_sim_use_trail",
    "get_sim_window_minutes",
    "load_sim_config",
    "resolve_sim_universe_path",
]
