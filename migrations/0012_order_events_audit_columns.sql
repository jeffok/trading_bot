-- Iter10: add audit columns to order_events (V8.3 recommended fields)
ALTER TABLE order_events
  ADD COLUMN IF NOT EXISTS action VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS actor VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS event_ts_hk TIMESTAMP NULL;

-- Helpful indexes for audit queries
CREATE INDEX IF NOT EXISTS idx_order_events_trace_time ON order_events(trace_id, created_at);
CREATE INDEX IF NOT EXISTS idx_order_events_action_time ON order_events(action, created_at);
