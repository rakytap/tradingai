from pathlib import Path


def test_schema_file_exists_and_defines_core_tables():
    schema = Path("src/tradingai/data/schema.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS stocks" in schema
    assert "CREATE TABLE IF NOT EXISTS daily_prices" in schema
    assert "CREATE TABLE IF NOT EXISTS news_events" in schema
    assert "CREATE TABLE IF NOT EXISTS predictions" in schema
