# Trading AI

A Python MCP server suite that exposes stock technical-analysis, news/sentiment, market-regime, and predictive signal-fusion tools to Cursor.

## Features

- Pluggable market data providers (`yfinance` today, Alpaca stub for later)
- Technical MCP tools for OHLCV data, indicator calculation, setup analysis, and multi-symbol comparison
- Separate news/sentiment MCP for headlines, SEC filings, earnings materials, analyst actions, and structured sentiment reports
- Separate market-regime MCP for bull/bear/sideways/high-volatility detection from macro features
- Separate predictive MCP that fuses technical, news, and regime signals into deterministic multi-horizon probabilities with SQLite-backed evaluation
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

1. Restart Cursor or toggle the `tradingai`, `tradingai-news`, `tradingai-regime`, and `tradingai-predict` MCP servers in **Customize**
2. Check **Output > MCP Logs** if a server does not appear
3. Ask Cursor to use the tools, for example:

- "Use tradingai to analyze the technical setup for AAPL over 6 months."
- "Compare technical signals for AAPL, MSFT, and GOOGL."
- "Use tradingai-news to analyze sentiment for NVDA and return catalysts and risks."
- "Use tradingai-regime to detect the current market regime and recommended strategy."
- "Use tradingai-predict to generate a multi-horizon prediction for AAPL."

## MCP Servers

### `tradingai` — Technical Analysis

| Tool | Description |
|------|-------------|
| `get_ohlcv` | Fetch normalized OHLCV history |
| `calculate_indicators` | Compute RSI, MACD, SMA/EMA, Bollinger Bands, ATR, momentum, volatility, volume anomalies, relative strength |
| `analyze_technical_setup` | Return trend/momentum/volatility signals plus a summary |
| `compare_technical_signals` | Compare technical bias across multiple tickers |

### `tradingai-news` — News / Sentiment

| Tool | Description |
|------|-------------|
| `fetch_company_news` | Recent headlines from enabled providers |
| `fetch_press_releases` | Press releases and IR announcements |
| `fetch_sec_filings` | Recent SEC 8-K, 10-Q, and 10-K filings |
| `fetch_earnings_materials` | Latest earnings press release and transcript when available |
| `fetch_analyst_actions` | Analyst upgrades, downgrades, and recommendation changes |
| `fetch_social_sentiment` | Filtered Reddit/X posts with social-only sentiment score |
| `analyze_news_sentiment` | Structured sentiment report with catalysts and risk factors |

### `tradingai-regime` — Market Regime

| Tool | Description |
|------|-------------|
| `fetch_macro_features` | Raw and derived macro feature values (VIX, SPY trend, yields, inflation, credit) |
| `detect_market_regime` | Primary tool — regime label, recommended strategy, confidence, and signal breakdown |
| `explain_regime` | Human-readable explanation of the current regime |

### `tradingai-predict` — Predictive Signal Fusion

| Tool | Description |
|------|-------------|
| `get_agent_signals` | Normalized technical, news, and regime signals for a ticker |
| `generate_prediction` | Primary tool — fuse signals, compute 1/5/20-day probabilities, auto-save to SQLite |
| `explain_prediction` | Human-readable evidence summary with deterministic probability numbers |
| `evaluate_predictions` | Backfill actual returns for mature predictions |
| `get_evaluation_metrics` | Accuracy, precision/recall, MAE, Brier score, top-10 ranked return |
| `get_prediction_history` | Query stored predictions for a ticker or date range |
| `get_prediction_track_record` | Segment hit rate, average return, and overconfidence |
| `get_segment_calibration` | Accuracy and Brier score by sector and prediction type |

## Configuration

Set providers in `.env`:

```bash
TRADINGAI_DATA_PROVIDER=yfinance
TRADINGAI_NEWS_PROVIDERS=yfinance,sec_edgar,rss
TRADINGAI_SOCIAL_PROVIDERS=reddit
TRADINGAI_SOCIAL_WEIGHT=0.45
SEC_IDENTITY="Your Name your@email.com"
REDDIT_USER_AGENT="tradingai-news-mcp/0.1 by tradingai"
TRADINGAI_REGIME_PROVIDERS=yfinance,fred
TRADINGAI_PREDICT_WEIGHTS=0.45,0.35,0.20
TRADINGAI_PREDICT_HORIZONS=1,5,20
TRADINGAI_DATABASE_URL=postgresql://localhost:5432/tradingai
TRADINGAI_SYNC_TICKERS=AAPL,MSFT,NVDA,AMD,INTC,GOOGL,META,TSLA,SPY
```

## PostgreSQL learning pipeline

Predictions, prices, news, and outcomes are stored in PostgreSQL for segment-level learning.

### Conda Postgres setup

```bash
conda activate stocks
conda install -c conda-forge postgresql psycopg
bash scripts/pg-init.sh
```

### Data sync commands

```bash
uv run tradingai-sync-stocks
uv run tradingai-sync-prices
uv run tradingai-sync-news
```

### Track-record example prompt

- "What is our hit rate for earnings-driven rallies in semiconductor stocks over 5 days?"

Use `get_prediction_track_record` and `get_segment_calibration` from `tradingai-predict`.

### Daily evaluation cron

```bash
0 18 * * 1-5  uv run tradingai-sync-prices && uv run tradingai-predict-eval
```

Optional one-time SQLite migration:

```bash
python scripts/migrate_sqlite_to_postgres.py
```

Optional providers:

- `newsapi` — requires `NEWS_API_KEY`
- `finnhub` — requires `FINNHUB_API_KEY`
- `x` — requires `X_BEARER_TOKEN` (X API v2 bearer token)
- `reddit` — uses public JSON API; if blocked, set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`
- `fred` — requires `FRED_API_KEY` (free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)); enriches yield curve, inflation, and credit spread signals

Social sentiment notes:

- Reddit is enabled by default; uses public API first, then PRAW credentials if configured
- X is optional and only activates when `x` is listed in `TRADINGAI_SOCIAL_PROVIDERS` and `X_BEARER_TOKEN` is set
- Social posts are noise-filtered (spam, memes, low engagement, missing ticker mentions)
- Social sentiment is weighted lower than news/filings via `TRADINGAI_SOCIAL_WEIGHT` (default `0.45`)

Supported technical data providers:

- `yfinance` — free market data (default)
- `alpaca` — stub only for now

## Development

Run the MCP servers directly:

```bash
uv run tradingai-mcp
uv run tradingai-news-mcp
uv run tradingai-regime-mcp
uv run tradingai-predict-mcp
```

Run the daily evaluation loop (for cron/systemd):

```bash
uv run tradingai-predict-eval
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

## Disclaimer

This project provides technical analysis for informational purposes only. It is not financial advice.
