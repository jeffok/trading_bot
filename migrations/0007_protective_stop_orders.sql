-- 0007_protective_stop_orders.sql
-- Milestone B1: store protective stop order ids for consistent stop-loss behavior.

ALTER TABLE trade_logs
  ADD COLUMN IF NOT EXISTS stop_client_order_id VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS stop_exchange_order_id VARCHAR(128) NULL,
  ADD COLUMN IF NOT EXISTS stop_order_type VARCHAR(32) NULL;

ALTER TABLE trade_logs_history
  ADD COLUMN IF NOT EXISTS stop_client_order_id VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS stop_exchange_order_id VARCHAR(128) NULL,
  ADD COLUMN IF NOT EXISTS stop_order_type VARCHAR(32) NULL;

-- Default runtime config (idempotent)
INSERT INTO system_config("key","value") VALUES
  ('USE_PROTECTIVE_STOP_ORDER','true'),
  ('STOP_ORDER_POLL_SECONDS','10')
ON CONFLICT ("key") DO NOTHING;
