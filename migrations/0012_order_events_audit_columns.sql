-- Iter10: add audit columns to order_events (V8.3 recommended fields)
ALTER TABLE order_events
  ADD COLUMN action VARCHAR(64) NULL AFTER event_type,
  ADD COLUMN actor VARCHAR(64) NULL AFTER action,
  ADD COLUMN event_ts_hk DATETIME NULL AFTER created_at;

-- Helpful indexes for audit queries
CREATE INDEX idx_order_events_trace_time ON order_events(trace_id, created_at);
CREATE INDEX idx_order_events_action_time ON order_events(action, created_at);
