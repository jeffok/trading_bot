-- 简单的SQL查询来诊断为什么没有下单

-- 1. 检查HALT状态
SELECT 'HALT_TRADING状态' AS check_type, `key`, `value`, updated_at 
FROM system_config 
WHERE `key` = 'HALT_TRADING';

SELECT 'EMERGENCY_EXIT状态' AS check_type, `key`, `value`, updated_at 
FROM system_config 
WHERE `key` = 'EMERGENCY_EXIT';

-- 2. 检查最近的订单事件（看是否有被拒绝的订单）
SELECT '最近订单事件' AS check_type, 
       event_type, 
       side, 
       status, 
       reason_code, 
       reason, 
       created_at,
       symbol
FROM order_events 
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY created_at DESC 
LIMIT 20;

-- 3. 检查当前持仓
SELECT '当前持仓' AS check_type, 
       symbol, 
       base_qty, 
       avg_entry_price,
       updated_at
FROM positions 
WHERE base_qty > 0;

-- 4. 检查市场数据缓存（最新K线）
SELECT '市场数据缓存' AS check_type,
       symbol,
       interval_minutes,
       open_time_ms,
       close_price,
       feature_version,
       updated_at,
       TIMESTAMPDIFF(SECOND, updated_at, NOW()) AS lag_seconds
FROM market_data_cache
ORDER BY updated_at DESC
LIMIT 20;

-- 5. 检查服务状态
SELECT '服务状态' AS check_type,
       service_name,
       instance_id,
       last_heartbeat,
       status_json,
       TIMESTAMPDIFF(SECOND, last_heartbeat, NOW()) AS heartbeat_lag_seconds
FROM service_status
ORDER BY last_heartbeat DESC
LIMIT 10;

-- 6. 检查最近的交易日志
SELECT '最近交易日志' AS check_type,
       symbol,
       side,
       qty,
       leverage,
       robot_score,
       ai_prob,
       open_reason_code,
       open_reason,
       created_at
FROM trade_logs
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY created_at DESC
LIMIT 10;
