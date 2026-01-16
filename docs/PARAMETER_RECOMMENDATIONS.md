# Bybit交易系统参数配置建议

## 一、立即调整的参数（高优先级）

### 1.1 策略参数（增加交易机会）

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

### 1.2 API请求频率参数（减少限流）

```bash
# 数据同步间隔（进一步减少API请求）
DATA_SYNC_LOOP_INTERVAL_SECONDS=60  # 从30增加到60秒
# 效果：20个交易对，每分钟约20次请求（原来40次）

# 交易对间延迟（保持）
DATA_SYNC_SYMBOL_DELAY_SECONDS=0.5  # 保持0.5秒
```

### 1.3 账户资金参数（必须设置）

```bash
# ⚠️ 重要：必须根据实际账户资金设置
ACCOUNT_EQUITY_USDT=你的实际账户权益  # 例如：1000, 5000, 10000等

# 说明：
# - 这个参数用于计算风险预算和订单大小
# - 如果设置错误，可能导致下单失败或风险计算错误
```

### 1.4 可选调整（根据资金量）

```bash
# 如果资金充足，可以增加并发持仓数
MAX_CONCURRENT_POSITIONS=5      # 从3增加到5（需要更多资金）

# 如果资金较少，保持3个或更少
# MAX_CONCURRENT_POSITIONS=3   # 默认值，适合小资金
```

---

## 二、参数详细说明

### 2.1 Setup B策略参数

| 参数 | 默认值 | 建议值 | 说明 |
|------|--------|--------|------|
| `SETUP_B_ADX_MIN` | 20 | 18 | ADX（平均趋向指标）阈值。18-20表示强趋势，15-18表示中等趋势 |
| `SETUP_B_VOL_RATIO_MIN` | 1.5 | 1.3 | 成交量比率。1.5倍表示明显放量，1.3倍表示温和放量 |
| `SETUP_B_AI_SCORE_MIN` | 55 | 50 | AI评分阈值。55分较保守，50分更平衡 |

**调整建议**：
- 如果长时间没有交易，先降低到建议值
- 观察1-2周，如果交易频率合适，保持
- 如果交易太频繁，可以适当提高

### 2.2 风控参数

| 参数 | 默认值 | 建议值 | 说明 |
|------|--------|--------|------|
| `RISK_BUDGET_PCT` | 0.03 (3%) | 0.03 | 单笔交易风险预算。3%是保守且合理的设置 |
| `MAX_DRAWDOWN_PCT` | 0.15 (15%) | 0.15 | 最大回撤限制。15%是合理的止损线 |
| `HARD_STOP_LOSS_PCT` | 0.03 (3%) | 0.03 | 硬止损比例。3%适合大多数情况 |
| `ACCOUNT_EQUITY_USDT` | 500 | **必须设置** | 账户权益。必须根据实际资金设置 |

**调整建议**：
- 风控参数建议保持默认值（较保守）
- 只有在有充分理由时才调整
- `ACCOUNT_EQUITY_USDT`必须准确设置

### 2.3 API请求频率参数

| 参数 | 默认值 | 建议值 | 说明 |
|------|--------|--------|------|
| `DATA_SYNC_LOOP_INTERVAL_SECONDS` | 30 | 60 | 数据同步主循环间隔。60秒更保守 |
| `DATA_SYNC_SYMBOL_DELAY_SECONDS` | 0.5 | 0.5 | 交易对间延迟。保持0.5秒 |
| `BYBIT_RECV_WINDOW` | 5000 | 5000 | 接收窗口。保持默认值 |

**调整建议**：
- 如果仍有速率限制问题，可以进一步增加间隔
- 监控API请求频率，确保不超过限制

### 2.4 交易执行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INTERVAL_MINUTES` | 15 | K线周期。15分钟适合趋势策略 |
| `MAX_CONCURRENT_POSITIONS` | 3 | 最大并发持仓数。根据资金量调整 |
| `MIN_ORDER_USDT` | 50 | 最小订单保证金。保持默认值 |
| `AUTO_LEVERAGE_MIN` | 10 | 自动杠杆下限。保持默认值 |
| `AUTO_LEVERAGE_MAX` | 20 | 自动杠杆上限。保持默认值 |

---

## 三、参数配置示例

### 3.1 保守配置（适合小资金，稳健策略）

```bash
# 策略参数（较严格）
SETUP_B_ADX_MIN=20
SETUP_B_VOL_RATIO_MIN=1.5
SETUP_B_AI_SCORE_MIN=55

# 风控参数
ACCOUNT_EQUITY_USDT=500
RISK_BUDGET_PCT=0.03
MAX_CONCURRENT_POSITIONS=3

# API频率（保守）
DATA_SYNC_LOOP_INTERVAL_SECONDS=60
```

**特点**：
- 交易频率较低
- 风险控制严格
- API请求较少

### 3.2 平衡配置（推荐，适合中等资金）

```bash
# 策略参数（平衡）
SETUP_B_ADX_MIN=18
SETUP_B_VOL_RATIO_MIN=1.3
SETUP_B_AI_SCORE_MIN=50

# 风控参数
ACCOUNT_EQUITY_USDT=2000
RISK_BUDGET_PCT=0.03
MAX_CONCURRENT_POSITIONS=5

# API频率（平衡）
DATA_SYNC_LOOP_INTERVAL_SECONDS=60
```

**特点**：
- 交易频率适中
- 风险控制合理
- API请求可控

### 3.3 激进配置（适合大资金，高频策略）

```bash
# 策略参数（较宽松）
SETUP_B_ADX_MIN=15
SETUP_B_VOL_RATIO_MIN=1.2
SETUP_B_AI_SCORE_MIN=45

# 风控参数
ACCOUNT_EQUITY_USDT=10000
RISK_BUDGET_PCT=0.03
MAX_CONCURRENT_POSITIONS=10

# API频率（需要WebSocket支持）
DATA_SYNC_LOOP_INTERVAL_SECONDS=30
```

**特点**：
- 交易频率较高
- 需要更严格的风控
- 建议使用WebSocket减少API请求

---

## 四、参数调整流程

### 步骤1：诊断当前状态
1. 检查HALT状态
2. 查看最近的SETUP_B_REJECT日志
3. 分析API请求频率
4. 检查账户资金设置

### 步骤2：调整策略参数
1. 先降低一个参数（如ADX_MIN）
2. 观察1-2天
3. 如果有效果，保持；如果没效果，继续调整
4. 逐步调整其他参数

### 步骤3：优化API频率
1. 增加数据同步间隔
2. 监控速率限制错误
3. 如果仍有问题，进一步增加间隔

### 步骤4：验证和监控
1. 观察交易执行情况
2. 监控API请求频率
3. 检查错误日志
4. 根据实际情况微调

---

## 五、参数监控建议

### 5.1 关键指标监控

1. **API请求频率**
   - 每分钟总请求数
   - 每个budget的请求数
   - 速率限制错误频率

2. **交易执行情况**
   - 下单成功率
   - Setup B条件满足率
   - 订单成交时间

3. **策略表现**
   - 交易频率
   - 盈亏情况
   - 最大回撤

### 5.2 告警设置

建议设置以下告警：
- API速率限制错误频率 > 5次/小时
- 数据延迟 > 5分钟
- 连续3个tick没有交易机会
- 账户回撤 > 10%

---

## 六、常见问题

### Q1: 参数调整后多久能看到效果？
A: 通常1-2个tick（15-30分钟）就能看到效果。但建议观察1-2天再决定是否继续调整。

### Q2: 如何知道参数是否合适？
A: 
- 交易频率：每天1-5笔交易较合理
- API错误：速率限制错误 < 1次/小时
- 策略表现：盈亏比 > 1.5

### Q3: 参数调整后风险会增加吗？
A: 降低策略阈值会增加交易频率，但风控参数（RISK_BUDGET_PCT等）保持不变，单笔风险不会增加。但总风险可能因为交易频率增加而增加。

### Q4: 如何平衡交易频率和API限制？
A: 
- 使用WebSocket获取市场数据（长期方案）
- 增加数据同步间隔（短期方案）
- 优化订单状态轮询（已实现）

---

## 七、快速参考

### 最小配置（必须设置）

```bash
ACCOUNT_EQUITY_USDT=你的实际资金
SETUP_B_ADX_MIN=18
SETUP_B_VOL_RATIO_MIN=1.3
SETUP_B_AI_SCORE_MIN=50
DATA_SYNC_LOOP_INTERVAL_SECONDS=60
```

### 完整推荐配置

```bash
# 策略
SETUP_B_ADX_MIN=18
SETUP_B_VOL_RATIO_MIN=1.3
SETUP_B_AI_SCORE_MIN=50

# 风控
ACCOUNT_EQUITY_USDT=2000
RISK_BUDGET_PCT=0.03
MAX_CONCURRENT_POSITIONS=5

# API频率
DATA_SYNC_LOOP_INTERVAL_SECONDS=60
DATA_SYNC_SYMBOL_DELAY_SECONDS=0.5

# 其他
INTERVAL_MINUTES=15
MAX_CONCURRENT_POSITIONS=5
```
