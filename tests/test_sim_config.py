from __future__ import annotations

from pathlib import Path

import pytest

from tradingai.sim.config import (
    DEFAULT_SIM_CONFIG_PATH,
    clear_sim_config_cache,
    get_query_latency_seconds,
    get_query_pace_seconds,
    get_scan_batch_size,
    get_sim_database_url,
    get_sim_dip_pct,
    get_sim_require_status,
    get_sim_start_cash,
    load_sim_config,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_sim_config_cache()
    yield
    clear_sim_config_cache()


def test_packaged_dip_sim_toml_loads():
    assert DEFAULT_SIM_CONFIG_PATH.is_file()
    cfg = load_sim_config()
    assert "database" in cfg
    assert "strategy" in cfg
    assert get_sim_dip_pct() == pytest.approx(4.9)
    assert get_sim_require_status() == ("reclaiming",)
    assert get_sim_start_cash() == pytest.approx(100_000.0)
    assert "tradingai_sim" in get_sim_database_url()
    assert get_query_latency_seconds() == pytest.approx(0.5)
    assert get_query_pace_seconds() == pytest.approx(10.0)
    assert get_scan_batch_size() == 20


def test_custom_config_path(tmp_path: Path):
    path = tmp_path / "custom.toml"
    path.write_text(
        """
[database]
url = "postgresql://localhost:5432/custom_sim"

[strategy]
dip_pct = 3.5
require_status = ["reclaiming", "stabilizing"]
start_cash = 50000

[ibkr_query]
query_latency_seconds = 1.0
query_pace_seconds = 15.0
scan_batch_size = 10
""",
        encoding="utf-8",
    )
    assert get_sim_dip_pct(config_path=str(path)) == pytest.approx(3.5)
    assert get_sim_require_status(config_path=str(path)) == (
        "reclaiming",
        "stabilizing",
    )
    assert get_sim_database_url(config_path=str(path)).endswith("custom_sim")
    assert get_query_latency_seconds(config_path=str(path)) == pytest.approx(1.0)
    assert get_query_pace_seconds(config_path=str(path)) == pytest.approx(15.0)
    assert get_scan_batch_size(config_path=str(path)) == 10
