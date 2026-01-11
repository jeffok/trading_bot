-- V8.3: order_events.raw_payload_json must be desensitized (脱敏)
ALTER TABLE order_events
  ADD COLUMN raw_payload_json JSON NULL AFTER reason;
