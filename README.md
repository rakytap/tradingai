# Trading AI

A Python MCP server that exposes stock technical-analysis tools to Cursor.

## Features

- Pluggable market data providers (`yfinance` today, Alpaca stub for later)
- MCP tools for OHLCV data, indicator calculation, technical setup analysis, and multi-symbol comparison
- stdio transport for local Cursor integration

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

With uv (recommended):

```bash
uv sync --extra dev
cp .env.example .env
```

With pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

If `uv` is not installed, update [`.cursor/mcp.json`](.cursor/mcp.json) to use the virtualenv entrypoint:

```json
"command": "${workspaceFolder}/.venv/bin/tradingai-mcp",
"args": []
```

## Cursor Integration

This repo includes [`.cursor/mcp.json`](.cursor/mcp.json). After installing dependencies:

1. Restart Cursor or toggle the `tradingai` MCP server in **Customize**
2. Check **Output > MCP Logs** if the server does not appear
3. Ask Cursor to use the trading tools, for example:

- "Use tradingai to analyze the technical setup for AAPL over 6 months."
- "Compare technical signals for AAPL, MSFT, and GOOGL."
- "Calculate momentum, volatility, volume anomaly, and relative strength vs SPY for NVDA."

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_ohlcv` | Fetch normalized OHLCV history |
| `calculate_indicators` | Compute RSI, MACD, SMA/EMA, Bollinger Bands, ATR, momentum, volatility, volume anomalies, relative strength |
| `analyze_technical_setup` | Return trend/momentum/volatility signals plus a summary |
| `compare_technical_signals` | Compare technical bias across multiple tickers |

## Configuration

Set the active provider in `.env`:

```bash
TRADINGAI_DATA_PROVIDER=yfinance
```

Supported provider names:

- `yfinance` — free market data (default)
- `alpaca` — stub only for now

## Development

Run the MCP server directly:

```bash
uv run tradingai-mcp
```

Run tests:

```bash
uv run pytest
```

Run integration tests that hit live market data:

```bash
uv run pytest -m integration
```

## Roadmap

- Alpaca provider for paper/live market data
- Portfolio and order tools
- Backtesting engine
- News and sentiment analysis

## Disclaimer

This project provides technical analysis for informational purposes only. It is not financial advice.
