-- 0010_feature_version_support.sql
-- V8.3: feature cache versioning (feature_version) for market_data_cache / precompute_tasks

-- 1) market_data_cache
ALTER TABLE market_data_cache
  ADD COLUMN IF NOT EXISTS feature_version INT NOT NULL DEFAULT 1;

-- change primary key to include feature_version
ALTER TABLE market_data_cache
  DROP CONSTRAINT IF EXISTS market_data_cache_pkey;

ALTER TABLE market_data_cache
  ADD PRIMARY KEY (symbol, interval_minutes, open_time_ms, feature_version);

-- 2) market_data_cache_history
ALTER TABLE market_data_cache_history
  ADD COLUMN IF NOT EXISTS feature_version INT NOT NULL DEFAULT 1;

ALTER TABLE market_data_cache_history
  DROP CONSTRAINT IF EXISTS market_data_cache_history_pkey;

ALTER TABLE market_data_cache_history
  ADD PRIMARY KEY (symbol, interval_minutes, open_time_ms, feature_version);

-- 3) precompute_tasks
ALTER TABLE precompute_tasks
  ADD COLUMN IF NOT EXISTS feature_version INT NOT NULL DEFAULT 1;

ALTER TABLE precompute_tasks
  DROP CONSTRAINT IF EXISTS precompute_tasks_pkey;

ALTER TABLE precompute_tasks
  ADD PRIMARY KEY (symbol, interval_minutes, open_time_ms, feature_version);

CREATE INDEX IF NOT EXISTS idx_precompute_fv_status
  ON precompute_tasks(feature_version, status, symbol, interval_minutes, open_time_ms);
