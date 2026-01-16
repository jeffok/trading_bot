-- Seed default system config values

INSERT INTO system_config ("key", "value") VALUES
  ('HALT_TRADING', 'false'),
  ('EMERGENCY_EXIT', 'false')
ON CONFLICT ("key") DO UPDATE SET "value" = EXCLUDED."value";
