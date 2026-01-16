# Trading Test Tool - 交易系统管理工具

⚠️ **重要：此工具只能在Docker容器中使用**

## 使用方式

所有命令都必须在Docker容器中执行：

```bash
docker compose exec execution python -m scripts.trading_test_tool <command> [args...]
```

## 命令列表

### 1. prepare - 准备检查
检查配置、服务状态等（等同于status命令）

```bash
docker compose exec execution python -m scripts.trading_test_tool prepare
docker compose exec execution python -m scripts.trading_test_tool prepare --max-age-seconds 120 --wait-seconds 30
```

### 2. status - 查看系统状态
查看数据库、Redis、市场数据缓存、系统开关等状态

```bash
docker compose exec execution python -m scripts.trading_test_tool status
```

### 3. diagnose - 诊断为什么没有下单
诊断指定交易对或所有交易对为什么没有下单

```bash
# 诊断所有交易对
docker compose exec execution python -m scripts.trading_test_tool diagnose

# 诊断指定交易对
docker compose exec execution python -m scripts.trading_test_tool diagnose --symbol BTCUSDT
```

### 4. check - 语法检查
检查所有Python文件的语法

```bash
docker compose exec execution python -m scripts.trading_test_tool check
```

### 5. halt - 暂停交易
暂停所有交易（写入 HALT_TRADING=true）

```bash
docker compose exec execution python -m scripts.trading_test_tool halt \
  --by admin \
  --reason-code ADMIN_HALT \
  --reason "系统维护"
```

### 6. resume - 恢复交易
恢复交易（写入 HALT_TRADING=false）

```bash
docker compose exec execution python -m scripts.trading_test_tool resume \
  --by admin \
  --reason-code ADMIN_RESUME \
  --reason "维护完成"
```

### 7. emergency-exit - 紧急退出
紧急退出所有持仓（写入 EMERGENCY_EXIT=true）

```bash
docker compose exec execution python -m scripts.trading_test_tool emergency-exit \
  --by admin \
  --reason-code EMERGENCY_EXIT \
  --reason "紧急情况" \
  --confirm-code <确认码>
```

### 8. set - 设置配置
设置系统配置项

```bash
docker compose exec execution python -m scripts.trading_test_tool set \
  <key> <value> \
  --by admin \
  --reason-code ADMIN_UPDATE_CONFIG \
  --reason "更新配置"
```

### 9. get - 获取配置
获取系统配置项的值

```bash
docker compose exec execution python -m scripts.trading_test_tool get HALT_TRADING
```

### 10. list - 列出配置
列出所有系统配置（支持前缀过滤）

```bash
# 列出所有配置
docker compose exec execution python -m scripts.trading_test_tool list

# 列出指定前缀的配置
docker compose exec execution python -m scripts.trading_test_tool list --prefix HALT
```

### 11. smoke-test - 链路自检
一键检查数据库、Redis、市场数据缓存是否正常

```bash
docker compose exec execution python -m scripts.trading_test_tool smoke-test
docker compose exec execution python -m scripts.trading_test_tool smoke-test --wait-seconds 120 --max-age-seconds 120
```

### 12. e2e-test - 端到端测试
实盘闭环测试（BUY->SELL->校验真实pnl_usdt）

```bash
docker compose exec execution python -m scripts.trading_test_tool e2e-test --yes
```

### 13. query - SQL查询（调试用）
执行SQL查询（仅用于调试）

```bash
docker compose exec execution python -m scripts.trading_test_tool query --sql "SELECT * FROM system_config LIMIT 10"
```

## 注意事项

1. ⚠️ 所有命令都必须在Docker容器中执行
2. ⚠️ 确保`.env`文件已正确配置PostgreSQL和Redis连接
3. ⚠️ 高危操作（如emergency-exit）可能需要确认码
4. ⚠️ 所有写操作都会记录审计日志和发送Telegram通知（如果配置了）

## 迁移说明

此工具从`tools/admin_cli`迁移而来，现在位于`scripts/trading_test_tool`目录，只能在Docker环境中使用。

所有小工具都已集成到此工具中，后续新增的小工具也应该添加到这里。
