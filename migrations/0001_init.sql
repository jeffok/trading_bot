
-- Alpha-Sniper-V8 (B-lite) - MariaDB schema (MVP)
-- IMPORTANT:
-- - All timestamps are stored in UTC.
-- - Use BIGINT milliseconds for exchange time where needed (e.g., kline open_time_ms).

CREATE TABLE IF NOT EXISTS schema_migrations (
  version VARCHAR(32) PRIMARY KEY,
  applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS system_config (
  `key` VARCHAR(128) PRIMARY KEY,
  `value` TEXT NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS config_audit (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  actor VARCHAR(64) NOT NULL,
  action VARCHAR(64) NOT NULL,
  cfg_key VARCHAR(128) NOT NULL,
  old_value TEXT NULL,
  new_value TEXT NULL,
  trace_id VARCHAR(64) NOT NULL,
  reason_code VARCHAR(64) NOT NULL,
  reason TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS service_status (
  service_name VARCHAR(64) NOT NULL,
  instance_id VARCHAR(64) NOT NULL,
  last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status_json JSON NOT NULL,
  PRIMARY KEY (service_name, instance_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS control_commands (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  command VARCHAR(64) NOT NULL,
  payload_json JSON NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'NEW',
  processed_at TIMESTAMP NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
  PRIMARY KEY (symbol, interval_minutes, open_time_ms),
  INDEX idx_market_data_close_time (symbol, interval_minutes, close_time_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS market_data_cache (
  symbol VARCHAR(32) NOT NULL,
  interval_minutes INT NOT NULL,
  open_time_ms BIGINT NOT NULL,
  ema_fast DECIMAL(28, 12) NULL,
  ema_slow DECIMAL(28, 12) NULL,
  rsi DECIMAL(28, 12) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (symbol, interval_minutes, open_time_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS order_events (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
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
  payload_json JSON NOT NULL,
  UNIQUE KEY uq_client_order (exchange, symbol, client_order_id),
  INDEX idx_order_events_symbol_time (symbol, created_at),
  INDEX idx_order_events_exchange_id (exchange, exchange_order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS trade_logs (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  trace_id VARCHAR(64) NOT NULL,
  symbol VARCHAR(32) NOT NULL,
  side VARCHAR(8) NOT NULL,
  qty DECIMAL(28, 12) NOT NULL,
  entry_price DECIMAL(28, 12) NULL,
  exit_price DECIMAL(28, 12) NULL,
  pnl DECIMAL(28, 12) NULL,
  features_json JSON NOT NULL,
  label INT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS position_snapshots (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  symbol VARCHAR(32) NOT NULL,
  base_qty DECIMAL(28, 12) NOT NULL,
  avg_entry_price DECIMAL(28, 12) NULL,
  meta_json JSON NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS archive_audit (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  table_name VARCHAR(64) NOT NULL,
  from_open_time_ms BIGINT NULL,
  to_open_time_ms BIGINT NULL,
  moved_rows BIGINT NOT NULL,
  trace_id VARCHAR(64) NOT NULL,
  status VARCHAR(16) NOT NULL,
  message TEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ai_models (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  model_name VARCHAR(64) NOT NULL,
  version VARCHAR(64) NOT NULL,
  metrics_json JSON NOT NULL,
  blob LONGBLOB NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
