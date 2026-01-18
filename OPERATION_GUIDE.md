# Trading Bot 操作指南

本文档提供交易系统的完整操作指南，包括 Web 管理界面、tbot 工具使用、诊断、参数配置和故障排除。

---

## 一、Web 管理界面使用指南

### 1.1 访问地址

**管理界面地址：**
```
http://localhost:9001/admin/ui
```

**如何查看地址：**
- 本地开发：`http://localhost:9001/admin/ui`
- 服务器部署：`http://<服务器IP>:9001/admin/ui`
- 如果配置了域名：`http://<域名>:9001/admin/ui`

### 1.2 访问方式

#### 方法 1：直接在浏览器打开（推荐）

1. 确保服务已启动：
   ```bash
   docker compose ps
   ```

2. 在浏览器中打开：
   ```
   http://localhost:9001/admin/ui
   ```

3. 输入管理员 Token：
   - 首次访问会提示输入 Token
   - Token 来自 `.env` 文件中的 `ADMIN_TOKEN` 配置
   - 查看 Token：`cat .env | grep ADMIN_TOKEN`

4. 或者通过 URL 参数传递：
   ```
   http://localhost:9001/admin/ui?token=YOUR_ADMIN_TOKEN
   ```

#### 方法 2：通过命令行检查服务状态

```bash
# 检查 API 服务是否运行
curl http://localhost:9001/health

# 查看服务日志（确认端口）
docker compose logs api-service | grep -i "uvicorn\|port\|start"
```

### 1.3 管理界面功能

Web 管理界面提供以下功能：

1. **系统状态查看**
   - 交易所配置
   - 交易对列表
   - 暂停/紧急退出状态
   - 持仓数量
   - 服务运行状态

2. **配置管理**
   - 更新策略参数（ADX、成交量比率、AI评分等）
   - 更新风控参数（账户权益、风险预算、最大回撤等）
   - 更新 AI 参数（AI权重、学习率等）
   - 所有参数都有说明和参考值提示

3. **控制操作**
   - 暂停交易
   - 恢复交易
   - 紧急退出（清仓）

### 1.4 认证说明

- **认证方式**：Bearer Token
- **Token 位置**：`.env` 文件中的 `ADMIN_TOKEN`
- **查看 Token**：
  ```bash
  docker compose exec api-service cat /app/.env | grep ADMIN_TOKEN
  # 或者
  cat .env | grep ADMIN_TOKEN
  ```

### 1.5 常见问题

**Q: 无法访问管理界面？**
- 检查服务是否运行：`docker compose ps`
- 检查端口是否暴露：确认 `docker-compose.yml` 中 `api-service` 的端口映射为 `9001:8080`
- 检查防火墙：确认端口 9001 未被阻止

**Q: 提示 Token 无效？**
- 确认 `.env` 文件中的 `ADMIN_TOKEN` 配置正确
- 确认 URL 中的 token 参数正确（区分大小写）

**Q: 如何修改 Token？**
- 编辑 `.env` 文件，修改 `ADMIN_TOKEN`
- 重启服务：`docker compose restart api-service`

---

## 二、tbot 工具使用指南

### 1.1 工具简介

`tbot` 是交易系统的命令行管理工具，提供了完整的系统管理、诊断、配置和测试功能。所有操作都可以通过简单的命令完成，无需直接操作数据库或编写代码。

### 1.2 工具位置

- **文件位置**: `scripts/tbot`（项目根目录下的 `scripts/tbot`）
- **无需安装**: 工具已包含在项目中，可直接使用

### 1.3 使用方法（三种方式）

#### 方式 1：在 Docker 中使用（推荐）⭐

项目主要在 Docker 中运行，**推荐在 Docker 容器中使用**：

```bash
# 基本格式
docker compose exec api-service tbot <command> [args...]

# 示例：查看帮助
docker compose exec api-service tbot --help

# 示例：查看系统状态
docker compose exec api-service tbot status

# 示例：诊断问题
docker compose exec api-service tbot diagnose
```

#### 方式 2：使用脚本文件（本地开发）

在项目根目录执行：

```bash
# 使用脚本
./scripts/tbot --help
./scripts/tbot status
./scripts/tbot diagnose
```

#### 方式 3：使用 Python 模块（兼容方式）

```bash
# 在项目根目录
python -m scripts.trading_test_tool --help
python -m scripts.trading_test_tool status
```

> ⚠️ **重要提示**: 本文档中的示例都使用方式 1（Docker 方式）。如果使用其他方式，请将 `docker compose exec api-service tbot` 替换为 `./scripts/tbot` 或 `python -m scripts.trading_test_tool`。

### 1.4 查看帮助

```bash
# 查看所有可用命令
docker compose exec api-service tbot --help

# 查看具体命令的帮助信息
docker compose exec api-service tbot diagnose --help
docker compose exec api-service tbot backtest --help
docker compose exec api-service tbot restart --help
docker compose exec api-service tbot seed --help
```

### 1.5 常用命令列表

#### 系统状态查看

```bash
# 查看完整系统状态（数据库、Redis、市场数据缓存、系统开关等）
docker compose exec api-service tbot status

# 查看指定配置项
docker compose exec api-service tbot get HALT_TRADING
docker compose exec api-service tbot get EMERGENCY_EXIT

# 列出所有配置（支持前缀过滤）
docker compose exec api-service tbot list
docker compose exec api-service tbot list --prefix SETUP_B
```

#### 交易控制

```bash
# 暂停交易
docker compose exec api-service tbot halt \
  --by admin \
  --reason-code ADMIN_HALT \
  --reason "系统维护"

# 恢复交易
docker compose exec api-service tbot resume \
  --by admin \
  --reason-code ADMIN_RESUME \
  --reason "维护完成"

# 紧急退出（清仓所有持仓）
docker compose exec api-service tbot emergency-exit \
  --by admin \
  --reason-code EMERGENCY_EXIT \
  --reason "紧急情况" \
  --confirm-code <确认码>
```

#### 配置管理

```bash
# 设置配置项
docker compose exec api-service tbot set \
  SETUP_B_ADX_MIN 18 \
  --by admin \
  --reason-code ADMIN_UPDATE_CONFIG \
  --reason "降低ADX阈值增加交易机会"

# 获取配置项
docker compose exec api-service tbot get SETUP_B_ADX_MIN
```

#### 诊断和测试

```bash
# 诊断为什么没有下单（推荐）
docker compose exec api-service tbot diagnose

# 诊断指定交易对
docker compose exec api-service tbot diagnose --symbol BTCUSDT

# 链路自检（检查数据库、Redis、市场数据缓存）
docker compose exec api-service tbot smoke-test

# 端到端测试（实盘交易测试，需谨慎使用）
docker compose exec api-service tbot e2e-test --yes

# 历史回测（需要管理员Token）
docker compose exec api-service tbot backtest \
  --token YOUR_ADMIN_TOKEN \
  --symbol BTCUSDT \
  --months 6
```

#### 服务管理

```bash
# 重启单个服务
docker compose exec api-service tbot restart data-syncer
docker compose exec api-service tbot restart strategy-engine
docker compose exec api-service tbot restart api-service

# 重启所有服务
docker compose exec api-service tbot restart all

# 启用保护止损订单
docker compose exec api-service tbot arm-stop \
  --by admin \
  --reason-code ADMIN_UPDATE_CONFIG \
  --reason "启用保护止损" \
  --stop-poll-seconds 10
```

#### 数据工具

```bash
# 生成合成测试数据（用于测试）
docker compose exec api-service tbot seed --bars 260 --start-price 40000

# SQL查询（调试用）
docker compose exec api-service tbot query --sql "SELECT * FROM system_config LIMIT 10"
```

### 1.6 命令详解

#### diagnose - 诊断工具

最常用的诊断命令，用于分析为什么系统没有下单：

```bash
# 诊断所有交易对
docker compose exec api-service tbot diagnose

# 诊断指定交易对
docker compose exec api-service tbot diagnose --symbol BTCUSDT
```

**功能**:
- 检查 HALT_TRADING 状态
- 检查市场数据是否完整
- 检查 Setup B 条件是否满足
- 检查风险预算是否充足
- 检查持仓数量限制
- 提供详细的诊断结果和解决建议

#### status - 查看系统状态

查看系统的完整状态信息：

```bash
docker compose exec api-service tbot status
```

**显示内容**:
- 数据库连接状态
- Redis 连接状态
- 市场数据缓存状态
- 系统配置开关（HALT_TRADING、EMERGENCY_EXIT 等）
- 服务心跳状态

#### backtest - 历史回测

分析过去指定月数内 Setup B 信号出现次数：

```bash
docker compose exec api-service tbot backtest \
  --token YOUR_ADMIN_TOKEN \
  --symbol BTCUSDT \
  --months 6 \
  --interval 15
```

**参数说明**:
- `--token`: 管理员Token（必须，用于鉴权）
- `--symbol`: 交易对符号（默认：BTCUSDT）
- `--months`: 回测月数（默认：6个月）
- `--interval`: K线周期（分钟，默认使用配置中的值）

**功能**:
- 检查历史K线数据完整性
- 如果数据不足，自动从Bybit获取历史K线（支持分批次）
- 计算特征指标并存储
- 分析满足Setup B条件的信号次数
- 生成详细回测报告

#### restart - 重启服务

重启指定的服务或所有服务：

```bash
# 重启单个服务
docker compose exec api-service tbot restart data-syncer
docker compose exec api-service tbot restart strategy-engine
docker compose exec api-service tbot restart api-service

# 重启所有服务
docker compose exec api-service tbot restart all
```

### 1.7 使用建议

1. **日常操作**: 使用 `tbot status` 定期检查系统状态
2. **问题诊断**: 遇到问题时，首先使用 `tbot diagnose` 进行诊断
3. **参数调整**: 使用 `tbot set` 和 `tbot get` 管理配置，避免直接修改数据库
4. **安全操作**: 重要操作（如紧急退出）需要确认码，确保操作安全

### 1.8 注意事项

- ⚠️ 所有写操作都会记录审计日志和发送Telegram通知（如果配置了）
- ⚠️ 高危操作（如 `emergency-exit`）可能需要确认码
- ⚠️ 确保在Docker环境中使用时，容器已正常运行
- ⚠️ `backtest` 命令需要提供有效的管理员Token

---

## 二、诊断为什么没有下单

### 快速检查方法

#### 方法1: 通过API检查（推荐）

如果API服务正在运行，可以通过以下方式检查：

```bash
# 检查健康状态（包括HALT状态）
curl http://localhost:9001/health | jq

# 检查详细状态（需要admin token）
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     http://localhost:9001/admin/status | jq
```

#### 方法2: 使用 tbot 工具诊断

```bash
# 诊断所有交易对
docker compose exec api-service tbot diagnose

# 诊断指定交易对
docker compose exec api-service tbot diagnose --symbol BTCUSDT
```

#### 方法3: 检查Docker日志

```bash
# 查看strategy-engine日志
docker-compose logs strategy-engine --tail 100 | grep -E "SETUP_B_REJECT|HALT|BUY|下单"

# 查看最近的日志
docker-compose logs strategy-engine --since 1h | grep -E "SETUP_B|reason"
```

### 常见原因和解决方法

#### 1. HALT_TRADING = true

**症状**: 交易被暂停

**解决方法**:
```bash
# 通过API恢复
curl -X POST http://localhost:9001/admin/resume \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "by": "admin",
    "reason_code": "ADMIN_RESUME",
    "reason": "恢复交易"
  }'

# 或通过CLI（仅在Docker中使用）
docker compose exec api-service tbot resume \
  --by admin \
  --reason-code ADMIN_RESUME \
  --reason "恢复交易"
```

#### 2. Setup B条件不满足

**症状**: 日志中有 "SETUP_B_REJECT" 或 "SetupB未满足"

**需要同时满足的条件**:
- ADX >= 20 (SETUP_B_ADX_MIN)
- +DI > -DI
- Squeeze释放 (前一根squeeze_status=1, 当前=0)
- 动量由负转正 (mom10从负转正)
- Volume ratio >= 1.5 (SETUP_B_VOL_RATIO_MIN)
- AI score >= 55 (SETUP_B_AI_SCORE_MIN)

**解决方法**: 
- 降低阈值（如果条件太严格）- 见下方"参数配置建议"
- 等待市场条件满足
- 检查市场数据是否正常更新

#### 3. 没有前一根K线数据

**症状**: Setup B需要前一根K线来判断squeeze_release和mom_flip

**解决方法**: 确保data-syncer正常运行，市场数据完整

#### 4. 市场数据延迟

**症状**: 数据滞后，导致信号判断不准确

**检查方法**:
```bash
# 使用 tbot 工具检查
docker compose exec api-service tbot query --sql "SELECT symbol, TIMESTAMPDIFF(SECOND, updated_at, NOW()) AS lag_seconds FROM market_data_cache ORDER BY updated_at DESC LIMIT 10"
```

**解决方法**: 检查data-syncer服务状态，确保正常运行

---

## 三、参数配置建议

### 3.1 策略参数（增加交易机会）

如果交易信号太少，可以降低Setup B条件阈值：

```bash
# 在.env文件中添加或修改以下参数

# Setup B条件阈值（降低以增加交易机会）
SETUP_B_ADX_MIN=18              # 从20降至18（ADX>=18表示中等强度趋势）
SETUP_B_VOL_RATIO_MIN=1.3       # 从1.5降至1.3（成交量放大1.3倍即可）
SETUP_B_AI_SCORE_MIN=50         # 从55降至50（AI评分50分即可入场）

# 说明：
# - 这些参数降低后会增加交易频率，需要确保风控到位
# - 建议逐步调整，观察效果后再决定是否进一步降低
```

### 3.2 API请求频率参数（减少限流）

如果遇到API限流问题，可以调整数据同步间隔：

```bash
# 数据同步间隔（进一步减少API请求）
DATA_SYNC_LOOP_INTERVAL_SECONDS=60  # 从30增加到60秒
# 效果：20个交易对，每分钟约20次请求（原来40次）

# 交易对间延迟（保持）
DATA_SYNC_SYMBOL_DELAY_SECONDS=0.5  # 保持0.5秒
```

### 3.3 账户资金参数（必须设置）

```bash
# ⚠️ 重要：必须根据实际账户资金设置
ACCOUNT_EQUITY_USDT=你的实际账户权益  # 例如：1000, 5000, 10000等

# 说明：
# - 这个参数用于计算风险预算和订单大小
# - 如果设置错误，可能导致下单失败或风险计算错误
```

### 3.4 并发持仓数（根据资金量）

```bash
# 如果资金充足，可以增加并发持仓数
MAX_CONCURRENT_POSITIONS=5      # 从3增加到5（需要更多资金）

# 如果资金较少，保持3个或更少
# MAX_CONCURRENT_POSITIONS=3   # 默认值，适合小资金
```

### 3.5 止损参数

```bash
# 硬止损百分比（默认3%）
HARD_STOP_LOSS_PCT=0.03         # 3%，表示入场价下跌3%时止损

# 使用交易所保护止损单（推荐）
USE_PROTECTIVE_STOP_ORDER=true  # 使用交易所止损单，更可靠

# 止损单轮询间隔
STOP_ORDER_POLL_SECONDS=10      # 每10秒检查一次止损单状态
```

### 3.6 AI参数

```bash
# 启用AI评分
AI_ENABLED=true                 # 启用AI模型评分

# AI权重（AI评分与机器人评分的权重）
AI_WEIGHT=0.35                  # 35% AI评分，65% 机器人评分

# AI学习率
AI_LR=0.05                      # 学习率，控制模型更新速度
AI_L2=0.000001                  # L2正则化系数
AI_MIN_SAMPLES=50               # 最少样本数，低于此数不使用AI
```

### 3.7 风控参数

```bash
# 风险预算百分比（每次交易最大风险）
RISK_BUDGET_PCT=0.03            # 3%，表示每次交易最多亏损账户权益的3%

# 最大回撤百分比（触发熔断）
MAX_DRAWDOWN_PCT=0.15           # 15%，回撤超过此值触发熔断

# 熔断器参数
CIRCUIT_WINDOW_SECONDS=600      # 熔断窗口（10分钟）
CIRCUIT_RATE_LIMIT_THRESHOLD=8  # 速率限制阈值（窗口内超过8次限流触发熔断）
CIRCUIT_FAILURE_THRESHOLD=6     # 失败阈值（窗口内超过6次失败触发熔断）
```


---

## 四、故障排除

### 4.1 服务启动问题

**问题**: 服务无法启动或连接数据库失败

**检查**:
1. 检查数据库连接配置（`.env` 文件中的 `POSTGRES_URL`）
2. 检查数据库是否运行：`docker compose ps postgres`
3. 查看服务日志：`docker compose logs strategy-engine`

**解决**:
```bash
# 检查数据库状态
docker compose ps postgres

# 查看数据库日志
docker compose logs postgres

# 重启服务
docker compose restart strategy-engine
```

### 4.2 API限流问题

**问题**: 频繁出现速率限制错误（retCode=10006）

**检查**:
```bash
# 查看data-syncer日志
docker compose logs data-syncer | grep -i "rate_limit\|10006"

# 检查当前配置
docker compose exec api-service tbot get DATA_SYNC_LOOP_INTERVAL_SECONDS
```

**解决**:
1. 增加数据同步间隔（见"参数配置建议"）
2. 检查是否启用了WebSocket（`BYBIT_WS_ENABLED=true`）
3. 减少交易对数量（如果配置了太多交易对）

### 4.3 数据同步问题

**问题**: 市场数据延迟或缺失

**检查**:
```bash
# 查看data-syncer日志
docker compose logs data-syncer --tail 100

# 检查市场数据缓存
docker compose exec api-service tbot query --sql "SELECT symbol, MAX(updated_at) as last_update FROM market_data_cache GROUP BY symbol"
```

**解决**:
```bash
# 重启data-syncer
docker compose restart data-syncer

# 检查服务状态
docker compose exec api-service tbot status
```

### 4.4 订单问题

**问题**: 订单被拒绝或无法成交

**检查**:
```bash
# 查看最近的订单事件
docker compose exec api-service tbot query --sql "SELECT * FROM order_events ORDER BY created_at DESC LIMIT 20"

# 检查账户余额
# （需要通过API或交易所界面检查）
```

**解决**:
1. 检查账户余额是否充足
2. 检查杠杆设置是否合理
3. 检查风控参数（`RISK_BUDGET_PCT`）
4. 查看订单拒绝原因（`reason_code` 和 `reason`）

---

## 五、最佳实践

### 5.1 参数调整建议

1. **渐进式调整**: 不要一次性大幅调整多个参数，建议逐个调整并观察效果
2. **记录变更**: 每次参数调整都应该记录原因和效果，便于回溯
3. **测试验证**: 重要参数调整前，建议先在测试环境验证

### 5.2 监控建议

1. **定期检查日志**: 每天检查服务日志，关注错误和警告
2. **监控系统状态**: 使用 `tbot status` 定期检查系统状态
3. **跟踪交易记录**: 定期查看交易记录，分析盈亏情况

### 5.3 安全建议

1. **保护管理员Token**: 不要在日志或代码中暴露 `ADMIN_TOKEN`
2. **定期备份数据库**: 重要数据应定期备份
3. **监控异常活动**: 关注异常的交易活动或配置变更

---

## 六、技术支持

如果遇到问题，可以：

1. 查看本文档的故障排除部分
2. 使用 `tbot diagnose` 工具诊断问题
3. 查看服务日志：`docker compose logs <service-name>`
4. 检查系统状态：`tbot status`

---

*最后更新时间: 2024年*
