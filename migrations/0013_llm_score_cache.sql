-- Iter13: create LLM score cache table for persistent storage
CREATE TABLE IF NOT EXISTS llm_score_cache (
    cache_key VARCHAR(512) PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    direction VARCHAR(8) NOT NULL,
    score FLOAT NOT NULL,
    features_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_llm_score_cache_symbol_direction ON llm_score_cache(symbol, direction);
CREATE INDEX IF NOT EXISTS idx_llm_score_cache_updated_at ON llm_score_cache(updated_at);

-- Add comment
COMMENT ON TABLE llm_score_cache IS 'Persistent cache for LLM-based AI scores to avoid redundant API calls';
