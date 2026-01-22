-- Iter14: add cutoff_days column to archive_audit table
ALTER TABLE archive_audit
  ADD COLUMN IF NOT EXISTS cutoff_days INT NULL;

-- Add comment
COMMENT ON COLUMN archive_audit.cutoff_days IS 'Number of days used as cutoff for archival (e.g., 90)';
