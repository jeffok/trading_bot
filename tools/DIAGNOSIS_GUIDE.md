# 为什么没有下单 - 完整诊断指南

## 问题概述

系统运行了几天，有20个交易对，但一直没有下单的情况。

## 已完成的代码改进

1. ✅ **添加了日志记录**: 当Setup B条件不满足时，会记录 "SETUP_B_REJECT" 日志，包含详细原因
2. ✅ **创建了诊断工具**: `tools/diagnose_no_orders.py` (需要Python环境)
3. ✅ **创建了SQL查询脚本**: `tools/check_no_orders_simple.sql`

## 立即检查步骤

### 步骤1: 检查HALT状态（最重要）

**方法A: 通过API**
```bash
curl http://YOUR_API_HOST:8080/health | jq '.halt_trading'
```

**方法B: 直接查询数据库**
```sql
SELECT `key`, `value`, updated_at 
FROM system_config 
WHERE `key` = 'HALT_TRADING';
```

**如果返回 `true`，需要恢复交易**:
```bash
# 通过API
curl -X POST http://YOUR_API_HOST:8080/admin/resume \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"by": "admin", "reason_code": "ADMIN_RESUME", "reason": "恢复交易"}'

# 或通过CLI
python -m tools.admin_cli resume --by admin --reason-code ADMIN_RESUME --reason "恢复交易"
```

### 步骤2: 检查最近的订单事件

查看是否有被拒绝的订单或Setup B拒绝记录：

```sql
SELECT 
    event_type, 
    side, 
    status, 
    reason_code, 
    reason, 
    created_at,
    symbol
FROM order_events 
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
  AND (reason_code LIKE '%SETUP_B%' OR reason_code LIKE '%REJECT%')
ORDER BY created_at DESC 
LIMIT 50;
```

### 步骤3: 检查服务日志

查找 "SETUP_B_REJECT" 日志，了解具体哪些条件不满足：

```bash
# 如果使用Docker
docker logs strategy-engine 2>&1 | grep -E "SETUP_B_REJECT|SetupB未满足" | tail -50

# 或查看所有相关日志
docker logs strategy-engine 2>&1 | grep -E "SETUP_B|reason_code|should_buy" | tail -100
```

### 步骤4: 检查市场数据

确保市场数据正常更新：

```sql
SELECT 
    symbol,
    interval_minutes,
    open_time_ms,
    close_price,
    feature_version,
    updated_at,
    TIMESTAMPDIFF(SECOND, updated_at, NOW()) AS lag_seconds
FROM market_data_cache
WHERE symbol IN ('BTCUSDT', 'ETHUSDT')  -- 替换为你的交易对
ORDER BY updated_at DESC
LIMIT 20;
```

**如果lag_seconds > 300（5分钟），说明数据同步有问题**

### 步骤5: 检查Setup B条件

Setup B需要**同时满足**以下所有条件：

1. **ADX >= 20** (SETUP_B_ADX_MIN)
   - 检查: `features_json` 中的 `adx14` 字段

2. **+DI > -DI**
   - 检查: `plus_di14` > `minus_di14`

3. **Squeeze释放**
   - 需要前一根K线的 `squeeze_status = 1`
   - 当前K线的 `squeeze_status = 0`
   - ⚠️ **这需要前一根K线数据**

4. **动量由负转正**
   - 需要前一根K线的 `mom10 < 0`
   - 当前K线的 `mom10 > 0`
   - ⚠️ **这需要前一根K线数据**

5. **Volume ratio >= 1.5** (SETUP_B_VOL_RATIO_MIN)
   - 检查: `vol_ratio` 字段

6. **AI score >= 55** (SETUP_B_AI_SCORE_MIN)
   - 这个在AI选币阶段计算

**查询示例**:
```sql
-- 获取最新两根K线数据（用于判断squeeze_release和mom_flip）
SELECT 
    symbol,
    open_time_ms,
    close_price,
    features_json,
    updated_at
FROM market_data_cache
WHERE symbol = 'BTCUSDT'  -- 替换为你的交易对
  AND interval_minutes = 15
  AND feature_version = 1
ORDER BY open_time_ms DESC
LIMIT 2;
```

然后手动检查 `features_json` 中的字段。

## 常见问题和解决方案

### 问题1: Setup B条件太严格

**症状**: 日志显示 "SetupB未满足: adx<20, no_squeeze_release, no_mom_flip_pos" 等

**解决方案**: 降低阈值

```bash
# 在.env文件中添加或修改
SETUP_B_ADX_MIN=15          # 默认20，降低到15
SETUP_B_VOL_RATIO_MIN=1.2   # 默认1.5，降低到1.2
SETUP_B_AI_SCORE_MIN=50     # 默认55，降低到50
```

然后重启strategy-engine服务。

### 问题2: 缺少前一根K线数据

**症状**: Squeeze释放和动量翻转无法判断

**解决方案**: 
- 确保data-syncer正常运行
- 检查market_data_cache中是否有足够的历史数据
- 等待更多K线数据积累

### 问题3: AI选币未选中

**症状**: 即使满足Setup B，也需要被AI选币逻辑选中

**解决方案**: 
- 检查AI模型是否正常训练
- 检查机器人评分是否足够高
- 查看日志中的AI选币相关信息

### 问题4: 已达到最大并发持仓

**症状**: 已经有3个持仓（默认MAX_CONCURRENT_POSITIONS=3）

**解决方案**: 
- 等待现有持仓平仓
- 或增加 `MAX_CONCURRENT_POSITIONS` 环境变量

### 问题5: 风控拒绝

**症状**: 日志中有 "RISK_BUDGET_REJECT"

**解决方案**: 
- 检查账户权益是否足够
- 调整风险预算参数:
  - `RISK_BUDGET_PCT` (默认0.03)
  - `ACCOUNT_EQUITY_USDT` (默认500)

## 推荐的参数调整（如果条件太严格）

如果市场条件很难同时满足所有Setup B条件，可以适当降低阈值：

```bash
# 在.env文件中
SETUP_B_ADX_MIN=15              # 从20降低到15
SETUP_B_VOL_RATIO_MIN=1.2      # 从1.5降低到1.2  
SETUP_B_AI_SCORE_MIN=50        # 从55降低到50

# 可选：增加并发持仓数
MAX_CONCURRENT_POSITIONS=5     # 从3增加到5
```

**注意**: 降低阈值会增加交易频率，但也可能增加风险。建议逐步调整并观察效果。

## 监控建议

1. **定期检查日志**: 
   ```bash
   docker logs strategy-engine 2>&1 | grep "SETUP_B_REJECT" | tail -20
   ```

2. **监控市场数据延迟**:
   ```sql
   SELECT symbol, MAX(TIMESTAMPDIFF(SECOND, updated_at, NOW())) AS max_lag
   FROM market_data_cache
   GROUP BY symbol;
   ```

3. **检查服务状态**:
   ```bash
   curl http://YOUR_API_HOST:8080/health | jq
   ```

4. **分析历史满足条件的情况**:
   ```sql
   -- 查看哪些交易对曾经满足过条件（如果有历史数据）
   SELECT symbol, COUNT(*) as count
   FROM trade_logs
   WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
   GROUP BY symbol
   ORDER BY count DESC;
   ```

## 下一步行动

1. ✅ **立即检查HALT状态** - 这是最常见的原因
2. ✅ **查看最近的SETUP_B_REJECT日志** - 了解具体哪些条件不满足
3. ✅ **检查市场数据是否正常更新** - 确保data-syncer正常运行
4. ✅ **根据日志结果调整参数** - 如果条件太严格，适当降低阈值
5. ✅ **持续监控** - 观察调整后的效果

## 联系支持

如果以上步骤都无法解决问题，请提供：
1. HALT状态查询结果
2. 最近的SETUP_B_REJECT日志（至少20条）
3. 市场数据延迟情况
4. 服务健康状态
