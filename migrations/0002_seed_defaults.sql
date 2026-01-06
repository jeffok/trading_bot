
INSERT INTO system_config (`key`, `value`) VALUES
  ('HALT_TRADING', 'false'),
  ('EMERGENCY_EXIT', 'false')
ON DUPLICATE KEY UPDATE `value` = VALUES(`value`);
