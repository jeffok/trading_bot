-- Iter8: ai_models 增加 is_current 标记（用于启动加载/回滚）
ALTER TABLE ai_models
  ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_ai_models_current ON ai_models(model_name, is_current, id);
