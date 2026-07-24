import json
from pathlib import Path
from unittest.mock import patch

from tradingai.news.entity_graph import get_node, node_from_dict, resolve_ticker_node
from tradingai.news.kg_curator.apply import apply_proposal
from tradingai.news.kg_curator.validate import (
    sanitize_relation_list,
    validate_proposed_relations,
)


def test_resolve_llm_settings_groq(monkeypatch):
    from tradingai.news.kg_curator import config as cfg

    monkeypatch.setenv("TRADINGAI_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    cfg.clear_llm_config_cache()
    settings = cfg.resolve_llm_settings()
    assert settings["provider"] == "groq"
    assert settings["api_key"] == "gsk_test"
    assert "groq.com" in settings["base_url"]
    assert settings["model"] == "llama-3.3-70b-versatile"
    assert settings["key_env"] == "GROQ_API_KEY"
    cfg.clear_llm_config_cache()


def test_resolve_llm_settings_openai_default(monkeypatch):
    from tradingai.news.kg_curator import config as cfg

    monkeypatch.delenv("TRADINGAI_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    cfg.clear_llm_config_cache()
    settings = cfg.resolve_llm_settings()
    assert settings["provider"] == "openai"
    assert settings["api_key"] == "sk_test"
    assert settings["model"] == "gpt-4o-mini"
    cfg.clear_llm_config_cache()


def test_sanitize_drops_self_and_invalid():
    cleaned = sanitize_relation_list(
        ["MSFT", "nvda", "BADTICKERTOOLONG", "12", "AVGO", "nvda"],
        subject="MSFT",
    )
    assert cleaned == ["NVDA", "AVGO"]


def test_thematic_prefers_known_etfs():
    cleaned = sanitize_relation_list(
        ["SMH", "NOTREALETF", "XLU", "AAAAA"],
        subject="MSFT",
        thematic=True,
    )
    assert "SMH" in cleaned
    assert "XLU" in cleaned
    assert "NOTREALETF" not in cleaned


def test_merge_preserves_existing_peers():
    existing = {"peers": ["AAPL", "GOOGL"], "suppliers": ["NVDA"]}
    validated = validate_proposed_relations(
        "MSFT",
        {"peers": ["AMZN"], "suppliers": ["AVGO"], "customers": [], "thematic_proxies": ["SMH"]},
        existing=existing,
        mode="merge",
    )
    assert validated["proposed"]["peers"][:2] == ["AAPL", "GOOGL"]
    assert "AMZN" in validated["proposed"]["peers"]
    assert "NVDA" in validated["proposed"]["suppliers"]
    assert "AVGO" in validated["proposed"]["suppliers"]
    assert validated["proposed"]["thematic_proxies"] == ["SMH"]


def test_resolve_ticker_node_prefers_db(monkeypatch):
    static = get_node("MSFT")
    assert static is not None

    def fake_fetch(ticker, database_url=None):
        return {
            "ticker": "MSFT",
            "sector": "Technology",
            "peers": ["ZZZZ"],
            "suppliers": [],
            "customers": [],
            "thematic_proxies": ["SMH"],
            "commodities": [],
            "rate_sensitive": False,
            "fx_sensitive": False,
            "duration_bucket": None,
            "notes": "overlay",
            "raw": {"source": "llm_curator"},
        }

    monkeypatch.setattr(
        "tradingai.data.repositories.ticker_kg.fetch_ticker_kg_row",
        fake_fetch,
    )
    node = resolve_ticker_node("MSFT")
    assert node is not None
    assert node.peers == ("ZZZZ",)
    assert node.thematic_proxies == ("SMH",)


def test_resolve_falls_back_on_db_error(monkeypatch):
    def boom(ticker, database_url=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "tradingai.data.repositories.ticker_kg.fetch_ticker_kg_row",
        boom,
    )
    node = resolve_ticker_node("MSFT")
    assert node is not None
    assert "NVDA" in node.suppliers


def test_propose_parses_mocked_llm(tmp_path, monkeypatch):
    from tradingai.news.kg_curator import propose as propose_mod

    monkeypatch.setattr(
        propose_mod,
        "gather_evidence",
        lambda ticker: {
            "ticker": "MSFT",
            "stock": {"sector": "Technology"},
            "existing_node": get_node("MSFT").to_dict(),
            "headlines": ["MSFT boosts AI capex"],
            "filing_excerpts": [],
            "evidence_digest": ["AI capex"],
        },
    )
    monkeypatch.setattr(
        propose_mod,
        "chat_completion_json",
        lambda evidence, model=None, provider=None, **kwargs: (
            {
                "peers": ["ORCL"],
                "suppliers": ["NVDA", "FAKEFAKE"],
                "customers": [],
                "thematic_proxies": ["SMH"],
                "rationale": {"NVDA": "GPUs", "ORCL": "cloud peer"},
            },
            "mock-model",
        ),
    )
    proposal = propose_mod.propose_ticker_graph_edges(
        "MSFT",
        proposal_dir=tmp_path,
    )
    assert proposal["status"] == "pending"
    assert proposal["path"]
    assert "ORCL" in proposal["proposed"]["peers"]
    assert "FAKEFAKE" not in proposal["proposed"]["suppliers"]
    assert Path(proposal["path"]).exists()


def test_apply_proposal_writes_overlay(tmp_path, monkeypatch):
    proposal = {
        "ticker": "MSFT",
        "mode": "merge",
        "proposed": {
            "peers": ["ORCL"],
            "suppliers": ["NVDA"],
            "customers": [],
            "thematic_proxies": ["SMH"],
        },
        "rationale": {"ORCL": "cloud"},
        "model": "mock",
        "status": "pending",
    }
    path = tmp_path / "MSFT_test.json"
    path.write_text(json.dumps(proposal))

    written = {}

    def fake_upsert(node, database_url=None, preserve_overlays=False, raw_json=None):
        written["node"] = node
        written["raw"] = raw_json

    monkeypatch.setattr(
        "tradingai.news.kg_curator.apply.initialize_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        "tradingai.news.kg_curator.apply.upsert_ticker_kg_node",
        fake_upsert,
    )
    monkeypatch.setattr(
        "tradingai.news.kg_curator.apply.fetch_ticker_kg_row",
        lambda ticker, database_url=None: written["node"].to_dict(),
    )

    result = apply_proposal(path)
    assert result["status"] == "applied"
    assert "ORCL" in written["node"].peers
    assert written["raw"]["source"] == "llm_curator"
    saved = json.loads(path.read_text())
    assert saved["status"] == "applied"


def test_node_from_dict_roundtrip():
    node = node_from_dict(
        {
            "ticker": "msft",
            "sector": "Technology",
            "peers": ["aapl"],
            "suppliers": ["nvda"],
            "thematic_proxies": ["smh"],
        }
    )
    assert node.ticker == "MSFT"
    assert node.peers == ("AAPL",)
    assert node.suppliers == ("NVDA",)
