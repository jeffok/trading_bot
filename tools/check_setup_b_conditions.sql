-- 检查Setup B条件是否满足（需要手动分析）

-- 获取最新的市场数据特征
SELECT 
    symbol,
    interval_minutes,
    open_time_ms,
    close_price,
    features_json,
    feature_version,
    updated_at
FROM market_data_cache
WHERE feature_version = 1  -- 根据实际版本调整
ORDER BY symbol, open_time_ms DESC;

-- 分析features_json中的关键指标：
-- - adx14: 需要 >= 20
-- - plus_di14: 需要 > minus_di14
-- - squeeze_status: 需要前一根=1，当前=0（squeeze释放）
-- - mom10: 需要前一根<0，当前>0（动量转正）
-- - vol_ratio: 需要 >= 1.5
-- - AI评分: 需要 >= 55（这个在trade_logs或AI模型中）
