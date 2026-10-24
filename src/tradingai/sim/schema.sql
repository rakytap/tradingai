-- Dedicated schema for the Alpaca 1m open-dip simulator (tradingai_sim).

CREATE TABLE IF NOT EXISTS sim_bars_1m (
    ticker TEXT NOT NULL,
    bar_time TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (ticker, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_sim_bars_1m_time
    ON sim_bars_1m (bar_time);

CREATE TABLE IF NOT EXISTS sim_daily_closes (
    ticker TEXT NOT NULL,
    session_day DATE NOT NULL,
    prior_close DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ticker, session_day)
);

CREATE TABLE IF NOT EXISTS sim_runs (
    run_id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    start_cash DOUBLE PRECISION NOT NULL,
    end_cash DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    trade_count INTEGER NOT NULL DEFAULT 0,
    win_count INTEGER NOT NULL DEFAULT 0,
    summary JSONB
);

CREATE TABLE IF NOT EXISTS sim_orders (
    order_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES sim_runs (run_id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    status TEXT NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    limit_price DOUBLE PRECISION,
    stop_price DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL,
    filled_at TIMESTAMPTZ,
    fill_price DOUBLE PRECISION,
    position_id BIGINT
);

CREATE INDEX IF NOT EXISTS idx_sim_orders_run
    ON sim_orders (run_id);

CREATE TABLE IF NOT EXISTS sim_fills (
    fill_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES sim_runs (run_id) ON DELETE CASCADE,
    order_id BIGINT REFERENCES sim_orders (order_id) ON DELETE SET NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    filled_at TIMESTAMPTZ NOT NULL,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_sim_fills_run
    ON sim_fills (run_id);

CREATE TABLE IF NOT EXISTS sim_positions (
    position_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES sim_runs (run_id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_price DOUBLE PRECISION,
    exit_time TIMESTAMPTZ,
    exit_reason TEXT,
    realized_pnl DOUBLE PRECISION,
    limit_price DOUBLE PRECISION,
    stop_price DOUBLE PRECISION,
    trail_activate_price DOUBLE PRECISION,
    sell_trail DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_sim_positions_run
    ON sim_positions (run_id);
