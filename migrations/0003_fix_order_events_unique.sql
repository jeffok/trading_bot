-- Fix order_events unique key to preserve immutable event stream
-- Old: UNIQUE (exchange, symbol, client_order_id)  -> would drop later events
-- New: UNIQUE (exchange, symbol, client_order_id, event_type)

ALTER TABLE order_events
  DROP CONSTRAINT IF EXISTS uq_client_order;

ALTER TABLE order_events
  ADD CONSTRAINT uq_client_order_event UNIQUE (exchange, symbol, client_order_id, event_type);
