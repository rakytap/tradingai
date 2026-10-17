CREATE TABLE IF NOT EXISTS stocks (
    ticker TEXT PRIMARY KEY,
    sector TEXT,
    industry TEXT,
    market_cap BIGINT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS daily_prices (
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    date DATE NOT NULL,
    open NUMERIC(12,4),
    high NUMERIC(12,4),
    low NUMERIC(12,4),
    close NUMERIC(12,4),
    volume BIGINT,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date
    ON daily_prices(ticker, date);

CREATE TABLE IF NOT EXISTS news_events (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    timestamp TIMESTAMPTZ NOT NULL,
    headline TEXT NOT NULL,
    sentiment NUMERIC(5,4),
    impact TEXT,
    source TEXT,
    raw_json JSONB,
    UNIQUE(ticker, timestamp, headline)
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    timestamp TIMESTAMPTZ NOT NULL,
    horizon_days INT NOT NULL,

    probability_up NUMERIC(5,4) NOT NULL,
    expected_return_pct NUMERIC(8,4) NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    combined_score NUMERIC(8,4) NOT NULL,

    sector TEXT,
    regime TEXT,
    technical_score NUMERIC(8,4),
    news_score NUMERIC(8,4),
    regime_score NUMERIC(8,4),

    catalyst_tags TEXT[] NOT NULL DEFAULT '{}',
    reason_tags TEXT[] NOT NULL DEFAULT '{}',
    prediction_type TEXT,

    agent_payload JSONB NOT NULL DEFAULT '{}',
    reasons JSONB NOT NULL DEFAULT '[]',

    model_name TEXT,
    model_version TEXT,

    actual_return_pct NUMERIC(8,4),
    direction_correct BOOLEAN,
    evaluated_at TIMESTAMPTZ,

    UNIQUE(ticker, timestamp, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_predictions_sector_type
    ON predictions(sector, prediction_type);

CREATE INDEX IF NOT EXISTS idx_predictions_catalyst_tags
    ON predictions USING GIN(catalyst_tags);

CREATE INDEX IF NOT EXISTS idx_predictions_evaluated
    ON predictions(horizon_days, evaluated_at);

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_name TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS ml_rank_score NUMERIC(8,4);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS sector_rank INT;

CREATE TABLE IF NOT EXISTS regime_snapshots (
    id BIGSERIAL PRIMARY KEY,
    as_of TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_regime_snapshots_as_of
    ON regime_snapshots(as_of DESC);

CREATE TABLE IF NOT EXISTS news_graph_events (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL REFERENCES stocks(ticker),
    observed_at TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    polarity SMALLINT NOT NULL,
    themes TEXT[] NOT NULL DEFAULT '{}',
    sectors TEXT[] NOT NULL DEFAULT '{}',
    evidence TEXT,
    confidence NUMERIC(5,4),
    raw_json JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_news_graph_events_ticker_observed
    ON news_graph_events(ticker, observed_at DESC);

CREATE TABLE IF NOT EXISTS ticker_kg_nodes (
    ticker TEXT PRIMARY KEY REFERENCES stocks(ticker),
    sector TEXT,
    peers TEXT[] NOT NULL DEFAULT '{}',
    suppliers TEXT[] NOT NULL DEFAULT '{}',
    customers TEXT[] NOT NULL DEFAULT '{}',
    thematic_proxies TEXT[] NOT NULL DEFAULT '{}',
    commodities TEXT[] NOT NULL DEFAULT '{}',
    rate_sensitive BOOLEAN NOT NULL DEFAULT false,
    fx_sensitive BOOLEAN NOT NULL DEFAULT false,
    duration_bucket TEXT,
    notes TEXT,
    raw_json JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intraday_bars (
    ticker TEXT NOT NULL,
    bar_time TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    PRIMARY KEY (ticker, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_intraday_bars_ticker_time
    ON intraday_bars(ticker, bar_time DESC);
