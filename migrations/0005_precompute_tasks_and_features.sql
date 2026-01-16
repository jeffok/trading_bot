-- 0005_precompute_tasks_and_features.sql
-- Milestone D: precompute queue + richer feature cache payload (JSON)
-- PostgreSQL 16+ recommended.

-- 1) Add features_json to market_data_cache (non-breaking)
ALTER TABLE market_data_cache
  ADD COLUMN IF NOT EXISTS features_json JSONB NULL;

-- 2) Precompute task queue (idempotent)
CREATE TABLE IF NOT EXISTS precompute_tasks (
  symbol VARCHAR(32) NOT NULL,
  interval_minutes INT NOT NULL,
  open_time_ms BIGINT NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'PENDING', -- PENDING/DONE/ERROR
  try_count INT NOT NULL DEFAULT 0,
  last_error TEXT NULL,
  trace_id VARCHAR(64) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol, interval_minutes, open_time_ms)
);

CREATE INDEX idx_precompute_status ON precompute_tasks(status, symbol, interval_minutes, open_time_ms);

-- PostgreSQL: 使用触发器实现ON UPDATE CURRENT_TIMESTAMP
CREATE TRIGGER update_precompute_tasks_updated_at BEFORE UPDATE ON precompute_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 3) History table for market_data_cache (previous archive job expects it)
CREATE TABLE IF NOT EXISTS market_data_cache_history (
  symbol VARCHAR(32) NOT NULL,
  interval_minutes INT NOT NULL,
  open_time_ms BIGINT NOT NULL,
  ema_fast DECIMAL(28, 12) NULL,
  ema_slow DECIMAL(28, 12) NULL,
  rsi DECIMAL(28, 12) NULL,
  features_json JSONB NULL,
  created_at TIMESTAMP NOT NULL,
  archived_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol, interval_minutes, open_time_ms)
);

CREATE INDEX idx_mdc_hist_symbol_time ON market_data_cache_history(symbol, interval_minutes, open_time_ms);
CREATE INDEX idx_mdc_hist_archived_at ON market_data_cache_history(archived_at);
