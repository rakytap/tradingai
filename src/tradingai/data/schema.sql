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
