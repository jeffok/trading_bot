-- Alpha-Sniper-V8 (B-lite) - PostgreSQL schema (MVP)
-- 数据以JSONB格式存储，提供更好的灵活性和查询能力
-- IMPORTANT:
-- - All timestamps are stored in UTC.
-- - Use BIGINT milliseconds for exchange time where needed (e.g., kline open_time_ms).

CREATE TABLE IF NOT EXISTS schema_migrations (
  version VARCHAR(32) PRIMARY KEY,
  applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_config (
  "key" VARCHAR(128) PRIMARY KEY,
  "value" TEXT NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- PostgreSQL: 使用触发器实现ON UPDATE CURRENT_TIMESTAMP
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS config_audit (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  actor VARCHAR(64) NOT NULL,
  action VARCHAR(64) NOT NULL,
  cfg_key VARCHAR(128) NOT NULL,
  old_value TEXT NULL,
  new_value TEXT NULL,
  trace_id VARCHAR(64) NOT NULL,
  reason_code VARCHAR(64) NOT NULL,
  reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS service_status (
  service_name VARCHAR(64) NOT NULL,
  instance_id VARCHAR(64) NOT NULL,
  last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status_json JSONB NOT NULL,
  PRIMARY KEY (service_name, instance_id)
);

CREATE TABLE IF NOT EXISTS control_commands (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  command VARCHAR(64) NOT NULL,
  payload_json JSONB NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'NEW',
  processed_at TIMESTAMP NULL
);

CREATE TABLE IF NOT EXISTS market_data (
  symbol VARCHAR(32) NOT NULL,
  interval_minutes INT NOT NULL,
  open_time_ms BIGINT NOT NULL,
  close_time_ms BIGINT NOT NULL,
  open_price DECIMAL(28, 12) NOT NULL,
  high_price DECIMAL(28, 12) NOT NULL,
  low_price DECIMAL(28, 12) NOT NULL,
  close_price DECIMAL(28, 12) NOT NULL,
  volume DECIMAL(28, 12) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol, interval_minutes, open_time_ms)
);

CREATE INDEX idx_market_data_close_time ON market_data(symbol, interval_minutes, close_time_ms);

CREATE TABLE IF NOT EXISTS market_data_cache (
  symbol VARCHAR(32) NOT NULL,
  interval_minutes INT NOT NULL,
  open_time_ms BIGINT NOT NULL,
  ema_fast DECIMAL(28, 12) NULL,
  ema_slow DECIMAL(28, 12) NULL,
  rsi DECIMAL(28, 12) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol, interval_minutes, open_time_ms)
);

CREATE TABLE IF NOT EXISTS order_events (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  trace_id VARCHAR(64) NOT NULL,
  service VARCHAR(64) NOT NULL,
  exchange VARCHAR(16) NOT NULL,
  symbol VARCHAR(32) NOT NULL,
  client_order_id VARCHAR(64) NOT NULL,
  exchange_order_id VARCHAR(64) NULL,
  event_type VARCHAR(32) NOT NULL,
  side VARCHAR(8) NOT NULL,
  qty DECIMAL(28, 12) NOT NULL,
  price DECIMAL(28, 12) NULL,
  status VARCHAR(32) NOT NULL,
  reason_code VARCHAR(64) NOT NULL,
  reason TEXT NOT NULL,
  payload_json JSONB NOT NULL,
  CONSTRAINT uq_client_order UNIQUE (exchange, symbol, client_order_id)
);

CREATE INDEX idx_order_events_symbol_time ON order_events(symbol, created_at);
CREATE INDEX idx_order_events_exchange_id ON order_events(exchange, exchange_order_id);

CREATE TABLE IF NOT EXISTS trade_logs (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  trace_id VARCHAR(64) NOT NULL,
  symbol VARCHAR(32) NOT NULL,
  side VARCHAR(8) NOT NULL,
  qty DECIMAL(28, 12) NOT NULL,
  entry_price DECIMAL(28, 12) NULL,
  exit_price DECIMAL(28, 12) NULL,
  pnl DECIMAL(28, 12) NULL,
  features_json JSONB NOT NULL,
  label INT NULL
);

CREATE TABLE IF NOT EXISTS position_snapshots (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  symbol VARCHAR(32) NOT NULL,
  base_qty DECIMAL(28, 12) NOT NULL,
  avg_entry_price DECIMAL(28, 12) NULL,
  meta_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS archive_audit (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  table_name VARCHAR(64) NOT NULL,
  from_open_time_ms BIGINT NULL,
  to_open_time_ms BIGINT NULL,
  moved_rows BIGINT NOT NULL,
  trace_id VARCHAR(64) NOT NULL,
  status VARCHAR(16) NOT NULL,
  message TEXT NULL
);

CREATE TABLE IF NOT EXISTS ai_models (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  model_name VARCHAR(64) NOT NULL,
  version VARCHAR(64) NOT NULL,
  metrics_json JSONB NOT NULL,
  blob BYTEA NULL
);
