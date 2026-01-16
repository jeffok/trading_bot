-- V8.3: order_events.raw_payload_json must be desensitized (脱敏)
ALTER TABLE order_events
  ADD COLUMN IF NOT EXISTS raw_payload_json JSONB NULL;
