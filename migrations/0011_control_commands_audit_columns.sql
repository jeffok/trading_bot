-- 0011_control_commands_audit_columns.sql
-- V8.3: unify control_commands audit fields

ALTER TABLE control_commands
  ADD COLUMN IF NOT EXISTS trace_id VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS actor VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS reason_code VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS reason TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_control_commands_status_time
  ON control_commands(status, created_at);

CREATE INDEX IF NOT EXISTS idx_control_commands_trace
  ON control_commands(trace_id);
