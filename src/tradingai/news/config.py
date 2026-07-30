from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_NEWS_PROVIDERS = ("yfinance", "sec_edgar", "rss")
DEFAULT_SOCIAL_PROVIDERS = ("reddit",)
DEFAULT_SOCIAL_WEIGHT = 0.45
DEFAULT_NEWS_SCORING_PATH = Path("./data/news_scoring.json")
DEFAULT_EVENT_GRAPH_PATH = Path("./data/event_graph.json")
DEFAULT_L3_NEWS_SPILL_ALPHA = 0.25

DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "news": 1.0,
    "press": 1.1,
    "analyst": 1.4,
    "filings": 0.9,
    "earnings": 1.3,
    "transcript": 1.0,
    "social": DEFAULT_SOCIAL_WEIGHT,
}

DEFAULT_RECENCY: dict[str, float] = {
    "half_life_days": 7.0,
    "floor": 0.4,
    "missing_date_weight": 0.6,
}


def default_news_scoring_params() -> dict[str, Any]:
    return {
        "source_weights": dict(DEFAULT_SOURCE_WEIGHTS),
        "recency": dict(DEFAULT_RECENCY),
    }


@lru_cache
def get_news_providers() -> list[str]:
    raw = os.getenv("TRADINGAI_NEWS_PROVIDERS", ",".join(DEFAULT_NEWS_PROVIDERS))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


@lru_cache
def get_social_providers() -> list[str]:
    raw = os.getenv("TRADINGAI_SOCIAL_PROVIDERS", ",".join(DEFAULT_SOCIAL_PROVIDERS))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _env_social_weight() -> float:
    raw = os.getenv("TRADINGAI_SOCIAL_WEIGHT", str(DEFAULT_SOCIAL_WEIGHT))
    try:
        weight = float(raw)
    except ValueError:
        return DEFAULT_SOCIAL_WEIGHT
    return max(0.0, min(1.0, weight))


@lru_cache
def get_social_weight() -> float:
    params = get_news_scoring_params()
    social = params["source_weights"].get("social")
    if social is not None:
        return max(0.0, min(1.0, float(social)))
    return _env_social_weight()


@lru_cache
def is_news_scoring_enabled() -> bool:
    raw = os.getenv("TRADINGAI_NEWS_SCORING_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@lru_cache
def get_news_scoring_path() -> Path:
    raw = os.getenv("TRADINGAI_NEWS_SCORING_PATH", str(DEFAULT_NEWS_SCORING_PATH))
    return Path(raw)


@lru_cache
def load_news_scoring() -> dict | None:
    if not is_news_scoring_enabled():
        return None
    path = get_news_scoring_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        "source_weights" not in payload
        and "recency" not in payload
        and "l3_news_spill_alpha" not in payload
    ):
        return None
    return payload


def _merge_scoring_params(payload: dict | None) -> dict[str, Any]:
    params = default_news_scoring_params()
    params["source_weights"]["social"] = _env_social_weight()
    if not payload:
        return params

    source_weights = payload.get("source_weights")
    if isinstance(source_weights, dict):
        for key in DEFAULT_SOURCE_WEIGHTS:
            if key in source_weights:
                try:
                    params["source_weights"][key] = float(source_weights[key])
                except (TypeError, ValueError):
                    continue

    recency = payload.get("recency")
    if isinstance(recency, dict):
        for key in DEFAULT_RECENCY:
            if key in recency:
                try:
                    params["recency"][key] = float(recency[key])
                except (TypeError, ValueError):
                    continue
    return params


@lru_cache
def get_news_scoring_params() -> dict[str, Any]:
    return _merge_scoring_params(load_news_scoring())


def _read_news_scoring_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_news_scoring_params(result: dict, path: Path | None = None) -> Path:
    target = path or get_news_scoring_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_news_scoring_file(target)
    payload = {
        "tuned_at": result.get("tuned_at"),
        "metric": result.get("metric", "brier"),
        "source_weights": result["source_weights"],
        "recency": result["recency"],
        "sample_sizes": result.get("sample_sizes"),
        "baseline_validation": result.get("baseline_validation"),
        "tuned_validation": result.get("tuned_validation"),
    }
    if "l3_news_spill_alpha" in result:
        payload["l3_news_spill_alpha"] = max(
            0.0, min(1.0, float(result["l3_news_spill_alpha"]))
        )
    elif "l3_news_spill_alpha" in existing:
        try:
            payload["l3_news_spill_alpha"] = max(
                0.0, min(1.0, float(existing["l3_news_spill_alpha"]))
            )
        except (TypeError, ValueError):
            pass
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    clear_news_scoring_cache()
    return target


def save_l3_spill_alpha(
    alpha: float,
    *,
    path: Path | None = None,
    result: dict | None = None,
) -> Path:
    """Merge tuned L3 spill alpha into news_scoring.json without wiping other keys."""
    target = path or get_news_scoring_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_news_scoring_file(target)
    clipped = max(0.0, min(1.0, float(alpha)))
    payload["l3_news_spill_alpha"] = round(clipped, 4)
    if result:
        payload["l3_spill_tuned_at"] = result.get("tuned_at")
        if result.get("sample_sizes") is not None:
            payload["l3_spill_sample_sizes"] = result["sample_sizes"]
        if result.get("baseline_validation") is not None:
            payload["l3_spill_baseline_validation"] = result["baseline_validation"]
        if result.get("tuned_validation") is not None:
            payload["l3_spill_tuned_validation"] = result["tuned_validation"]
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    clear_news_scoring_cache()
    return target


def clear_news_scoring_cache() -> None:
    get_news_providers.cache_clear()
    get_social_providers.cache_clear()
    get_social_weight.cache_clear()
    is_news_scoring_enabled.cache_clear()
    get_news_scoring_path.cache_clear()
    load_news_scoring.cache_clear()
    get_news_scoring_params.cache_clear()
    clear_l3_spill_cache()


@lru_cache
def is_l3_news_spill_enabled() -> bool:
    raw = os.getenv("TRADINGAI_L3_NEWS_SPILL", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _alpha_from_env() -> float:
    raw = os.getenv("TRADINGAI_L3_NEWS_SPILL_ALPHA", str(DEFAULT_L3_NEWS_SPILL_ALPHA))
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except ValueError:
        return DEFAULT_L3_NEWS_SPILL_ALPHA


@lru_cache
def get_l3_news_spill_alpha() -> float:
    tuned = load_news_scoring()
    if tuned is not None and "l3_news_spill_alpha" in tuned:
        try:
            return max(0.0, min(1.0, float(tuned["l3_news_spill_alpha"])))
        except (TypeError, ValueError):
            pass
    return _alpha_from_env()


def clear_l3_spill_cache() -> None:
    is_l3_news_spill_enabled.cache_clear()
    get_l3_news_spill_alpha.cache_clear()


@lru_cache
def is_event_graph_enabled() -> bool:
    raw = os.getenv("TRADINGAI_EVENT_GRAPH_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@lru_cache
def get_event_graph_path() -> Path:
    raw = os.getenv("TRADINGAI_EVENT_GRAPH_PATH", str(DEFAULT_EVENT_GRAPH_PATH))
    return Path(raw)


@lru_cache
def load_event_graph() -> dict | None:
    if not is_event_graph_enabled():
        return None
    path = get_event_graph_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _merge_event_graph_params(payload: dict | None) -> dict[str, Any]:
    from tradingai.news.analysis.events import default_event_graph_params

    params = default_event_graph_params()
    if not payload:
        return params

    for key in ("propagation_decay", "score_divisor", "edge_scale"):
        if key in payload:
            try:
                params[key] = float(payload[key])
            except (TypeError, ValueError):
                pass
    if "max_hops" in payload:
        try:
            params["max_hops"] = int(payload["max_hops"])
        except (TypeError, ValueError):
            pass

    theme_edges = payload.get("theme_edges")
    if isinstance(theme_edges, dict) and theme_edges:
        merged_edges: dict[str, list[list]] = {}
        for theme, edges in theme_edges.items():
            if not isinstance(edges, list):
                continue
            parsed: list[list] = []
            for edge in edges:
                if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                    continue
                try:
                    parsed.append([str(edge[0]), float(edge[1])])
                except (TypeError, ValueError):
                    continue
            merged_edges[str(theme)] = parsed
        if merged_edges:
            params["theme_edges"] = merged_edges

    event_specs = payload.get("event_specs")
    if isinstance(event_specs, list) and event_specs:
        defaults_by_type = {
            str(spec["event_type"]): dict(spec) for spec in params["event_specs"]
        }
        merged_specs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in event_specs:
            if not isinstance(raw, dict) or not raw.get("event_type"):
                continue
            event_type = str(raw["event_type"])
            base = dict(defaults_by_type.get(event_type, {
                "event_type": event_type,
                "enabled": True,
                "polarity": 1,
                "themes": [],
                "patterns": [],
            }))
            if "enabled" in raw:
                base["enabled"] = bool(raw["enabled"])
            if "polarity" in raw:
                try:
                    base["polarity"] = int(raw["polarity"])
                except (TypeError, ValueError):
                    pass
            if isinstance(raw.get("themes"), list):
                base["themes"] = [str(t) for t in raw["themes"]]
            if isinstance(raw.get("patterns"), list) and raw["patterns"]:
                base["patterns"] = [str(p) for p in raw["patterns"] if p]
            merged_specs.append(base)
            seen.add(event_type)
        for event_type, spec in defaults_by_type.items():
            if event_type not in seen:
                merged_specs.append(spec)
        params["event_specs"] = merged_specs

    return params


@lru_cache
def get_event_graph_params() -> dict[str, Any]:
    return _merge_event_graph_params(load_event_graph())


def save_event_graph_params(result: dict, path: Path | None = None) -> Path:
    target = path or get_event_graph_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tuned_at": result.get("tuned_at"),
        "metric": result.get("metric", "brier"),
        "propagation_decay": result["propagation_decay"],
        "max_hops": result["max_hops"],
        "score_divisor": result["score_divisor"],
        "edge_scale": result["edge_scale"],
        "theme_edges": result["theme_edges"],
        "event_specs": result["event_specs"],
        "sample_sizes": result.get("sample_sizes"),
        "baseline_validation": result.get("baseline_validation"),
        "tuned_validation": result.get("tuned_validation"),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    clear_event_graph_cache()
    return target


def clear_event_graph_cache() -> None:
    is_event_graph_enabled.cache_clear()
    get_event_graph_path.cache_clear()
    load_event_graph.cache_clear()
    get_event_graph_params.cache_clear()


@lru_cache
def get_reddit_user_agent() -> str:
    return os.getenv(
        "REDDIT_USER_AGENT",
        "tradingai-news-mcp/0.1 by tradingai",
    )


@lru_cache
def get_reddit_client_id() -> str | None:
    return os.getenv("REDDIT_CLIENT_ID")


@lru_cache
def get_reddit_client_secret() -> str | None:
    return os.getenv("REDDIT_CLIENT_SECRET")


@lru_cache
def get_x_bearer_token() -> str | None:
    return os.getenv("X_BEARER_TOKEN")


@lru_cache
def get_sec_identity() -> str | None:
    return os.getenv("SEC_IDENTITY")


@lru_cache
def get_news_api_key() -> str | None:
    return os.getenv("NEWS_API_KEY")


@lru_cache
def get_finnhub_api_key() -> str | None:
    return os.getenv("FINNHUB_API_KEY")
