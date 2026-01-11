-- 0007_protective_stop_orders.sql
-- Milestone B1: store protective stop order ids for consistent stop-loss behavior.

ALTER TABLE trade_logs
  ADD COLUMN stop_client_order_id VARCHAR(64) NULL AFTER exchange_order_id,
  ADD COLUMN stop_exchange_order_id VARCHAR(128) NULL AFTER stop_client_order_id,
  ADD COLUMN stop_order_type VARCHAR(32) NULL AFTER stop_exchange_order_id;

ALTER TABLE trade_logs_history
  ADD COLUMN stop_client_order_id VARCHAR(64) NULL AFTER exchange_order_id,
  ADD COLUMN stop_exchange_order_id VARCHAR(128) NULL AFTER stop_client_order_id,
  ADD COLUMN stop_order_type VARCHAR(32) NULL AFTER stop_exchange_order_id;

-- Default runtime config (idempotent)
INSERT IGNORE INTO system_config(`key`,`value`) VALUES
  ('USE_PROTECTIVE_STOP_ORDER','true'),
  ('STOP_ORDER_POLL_SECONDS','10');
