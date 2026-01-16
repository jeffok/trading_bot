# 诊断为什么没有下单

## 快速检查方法

### 方法1: 通过API检查（推荐）

如果API服务正在运行，可以通过以下方式检查：

```bash
# 检查健康状态（包括HALT状态）
curl http://localhost:8080/health | jq

# 检查详细状态（需要admin token）
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     http://localhost:8080/admin/status | jq
```

### 方法2: 直接查询数据库

```bash
# 如果使用Docker（使用trading_test_tool）
docker compose exec execution python -m scripts.trading_test_tool query --sql "$(cat tools/check_no_orders_simple.sql)"

# 或者直接连接PostgreSQL
psql $POSTGRES_URL -f tools/check_no_orders_simple.sql
```

### 方法3: 检查Docker日志

```bash
# 查看strategy-engine日志
docker-compose logs strategy-engine --tail 100 | grep -E "SETUP_B_REJECT|HALT|BUY|下单"

# 查看最近的日志
docker-compose logs strategy-engine --since 1h | grep -E "SETUP_B|reason"
```

## 常见原因和解决方法

### 1. HALT_TRADING = true

**症状**: 交易被暂停

**解决方法**:
```bash
# 通过API恢复
curl -X POST http://localhost:8080/admin/resume \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "by": "admin",
    "reason_code": "ADMIN_RESUME",
    "reason": "恢复交易"
  }'

# 或通过CLI（仅在Docker中使用）
docker compose exec execution python -m scripts.trading_test_tool resume \
  --by admin \
  --reason-code ADMIN_RESUME \
  --reason "恢复交易"
```

### 2. Setup B条件不满足

**症状**: 日志中有 "SETUP_B_REJECT" 或 "SetupB未满足"

**需要同时满足的条件**:
- ADX >= 20 (SETUP_B_ADX_MIN)
- +DI > -DI
- Squeeze释放 (前一根squeeze_status=1, 当前=0)
- 动量由负转正 (mom10从负转正)
- Volume ratio >= 1.5 (SETUP_B_VOL_RATIO_MIN)
- AI score >= 55 (SETUP_B_AI_SCORE_MIN)

**解决方法**: 
- 降低阈值（如果条件太严格）
- 等待市场条件满足
- 检查市场数据是否正常更新

### 3. 没有前一根K线数据

**症状**: Setup B需要前一根K线来判断squeeze_release和mom_flip

**解决方法**: 
- 确保data-syncer正常运行
- 检查market_data_cache中是否有足够的历史数据

### 4. AI选币未选中

**症状**: 即使满足Setup B，也需要被AI选币逻辑选中

**解决方法**: 
- 检查AI模型是否正常
- 检查机器人评分是否足够高

### 5. 并发持仓限制

**症状**: 已达到最大并发持仓数（默认3个）

**解决方法**: 
- 等待现有持仓平仓
- 或增加 MAX_CONCURRENT_POSITIONS

### 6. 风控拒绝

**症状**: 日志中有 "RISK_BUDGET_REJECT"

**解决方法**: 
- 检查账户权益
- 调整风险预算参数

## 调整参数示例

如果Setup B条件太严格，可以降低阈值：

```bash
# 在.env文件中或环境变量中设置
export SETUP_B_ADX_MIN=15          # 降低ADX阈值
export SETUP_B_VOL_RATIO_MIN=1.2   # 降低成交量比率阈值
export SETUP_B_AI_SCORE_MIN=50     # 降低AI评分阈值
```

然后重启strategy-engine服务。

## 监控建议

1. **定期检查日志**: 关注 "SETUP_B_REJECT" 日志，了解具体哪些条件不满足
2. **监控市场数据**: 确保data-syncer正常运行，市场数据及时更新
3. **检查服务状态**: 定期查看 /health 或 /admin/status 接口
4. **分析历史数据**: 查看哪些交易对曾经满足过条件，了解市场特征
