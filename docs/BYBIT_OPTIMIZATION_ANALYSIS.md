# Bybit API 优化分析报告

## 执行摘要

本报告基于Bybit V5 API官方文档和当前代码实现，提供全面的架构优化建议和参数调整方案。

---

## 一、Bybit API限制规则分析

### 1.1 HTTP REST API限制

| 限制类型 | 规则 | 影响 |
|---------|------|------|
| **IP级别限制** | 每个IP每5秒最多600次请求 | 影响所有REST请求 |
| **UID级别限制** | 基于API Key的滚动窗口限制 | 影响签名请求 |
| **Endpoint级别限制** | 不同端点有不同限制：<br>- 订单创建：10-20次/秒<br>- 订单查询：50次/秒<br>- 市场数据：相对宽松 | 需要分类管理 |
| **retCode=10006** | "Too many visits! Exceeded the API Rate Limit" | 速率限制错误 |

### 1.2 WebSocket限制

| 限制类型 | 规则 | 建议 |
|---------|------|------|
| **连接数限制** | 每个IP 5分钟内最多500个连接 | 避免频繁重连 |
| **订阅数限制** | 单个连接可订阅多个topic | 合并订阅减少连接数 |
| **公共频道** | 无需认证，支持kline/ticker/orderbook | 优先使用 |

### 1.3 关键发现

1. **当前问题**：20个交易对 × 频繁轮询 = 容易触发限流
2. **优化方向**：使用WebSocket + 降低REST频率 + 智能限流

---

## 二、当前代码问题诊断

### 2.1 速率限制器配置问题

**当前配置** (`shared/exchange/bybit.py:112-114`):
```python
self.limiter.ensure_budget("market_data", 10, 20)  # 10 RPS, 20 burst
self.limiter.ensure_budget("account", 5, 10)       # 5 RPS, 10 burst
self.limiter.ensure_budget("order", 5, 10)         # 5 RPS, 10 burst
```

**问题分析**:
- ❌ `market_data` 预算过高：10 RPS对于20个交易对来说太激进
- ❌ 没有考虑Bybit的实际限制（订单创建10-20/s，但查询50/s）
- ❌ 没有区分公开API和私有API的限流策略

**建议优化**:
```python
# 公开市场数据：更保守（因为20个交易对）
self.limiter.ensure_budget("market_data", 2, 5)    # 2 RPS, 5 burst

# 账户查询：中等频率
self.limiter.ensure_budget("account", 3, 6)        # 3 RPS, 6 burst

# 订单操作：区分创建和查询
self.limiter.ensure_budget("order_create", 2, 4)   # 2 RPS, 4 burst（创建）
self.limiter.ensure_budget("order_query", 5, 10)   # 5 RPS, 10 burst（查询）
```

### 2.2 订单状态轮询问题

**当前实现** (`shared/exchange/bybit.py:447-468`):
```python
end = time.time() + 10.0
while time.time() < end:
    st = self.get_order_status(...)  # 每次调用都是REST请求
    # ...
    time.sleep(0.2)  # 200ms间隔
```

**问题分析**:
- ❌ 10秒内最多50次查询（0.2秒间隔），但每次都是独立REST请求
- ❌ 没有利用Bybit的WebSocket订单推送
- ❌ 如果订单已成交，仍会继续轮询直到超时

**优化建议**:
1. 增加初始延迟（订单创建后通常需要100-500ms才可查询）
2. 使用指数退避（0.2s → 0.5s → 1s）
3. 订单成交后立即退出循环

### 2.3 数据同步频率问题

**当前配置** (`services/data_syncer/main.py:930`):
```python
time.sleep(30)  # 30秒间隔
```

**问题分析**:
- ✅ 已优化为30秒（之前10秒）
- ⚠️ 20个交易对 × 0.5秒延迟 = 10秒处理时间
- ⚠️ 总循环约40秒，每分钟约30次请求（仍可能偏高）

**进一步优化**:
- 考虑增加到60秒（每分钟15次请求）
- 或实现WebSocket实时推送

### 2.4 签名和请求头问题

**当前实现** (`shared/exchange/bybit.py:121-124`):
```python
def _sign(self, payload: str, ts_ms: int) -> str:
    pre = f"{ts_ms}{self.api_key}{self.recv_window}{payload}"
    return hmac.new(self.api_secret, pre.encode("utf-8"), hashlib.sha256).hexdigest()
```

**分析**:
- ✅ 签名逻辑正确（符合Bybit V5规范）
- ✅ 已处理payload一致性（GET/POST分别处理）
- ✅ API密钥清理和验证已实现

**建议**:
- 考虑添加时间同步检查（NTP同步）
- 添加签名验证日志（仅开发环境）

### 2.5 recv_window配置

**当前配置** (`shared/config/loader.py:225`):
```python
bybit_recv_window: int = int(os.getenv("BYBIT_RECV_WINDOW", "5000"))
```

**分析**:
- 5000ms（5秒）是合理的默认值
- 如果网络延迟高，可以适当增加
- 如果时间同步不准确，需要增加

**建议**:
- 保持5000ms，除非有网络延迟问题
- 添加时间同步检查

---

## 三、参数指标分析与建议

### 3.1 交易策略参数

| 参数 | 当前值 | 分析 | 建议 |
|------|--------|------|------|
| `SETUP_B_ADX_MIN` | 20 | 中等严格，ADX>=20表示强趋势 | 可降至15-18，增加交易机会 |
| `SETUP_B_VOL_RATIO_MIN` | 1.5 | 合理，1.5倍成交量表示放量 | 保持或降至1.2-1.3 |
| `SETUP_B_AI_SCORE_MIN` | 55 | 较高，可能过于保守 | 建议降至50，观察效果 |
| `MAX_CONCURRENT_POSITIONS` | 3 | 保守，适合小资金 | 根据资金量调整（5-10） |
| `MIN_ORDER_USDT` | 50 | 合理的最小订单 | 保持 |
| `HARD_STOP_LOSS_PCT` | 0.03 (3%) | 合理的止损比例 | 保持 |

### 3.2 风控参数

| 参数 | 当前值 | 分析 | 建议 |
|------|--------|------|------|
| `RISK_BUDGET_PCT` | 0.03 (3%) | 保守，单笔风险3% | 保持，适合稳健策略 |
| `MAX_DRAWDOWN_PCT` | 0.15 (15%) | 合理的最大回撤 | 保持 |
| `ACCOUNT_EQUITY_USDT` | 500 | 需要根据实际资金调整 | **必须根据实际账户资金设置** |
| `AUTO_LEVERAGE_MIN` | 10 | 杠杆范围下限 | 保持 |
| `AUTO_LEVERAGE_MAX` | 20 | 杠杆范围上限 | 保持 |

### 3.3 API请求频率参数

| 参数 | 当前值 | 分析 | 建议 |
|------|--------|------|------|
| `DATA_SYNC_LOOP_INTERVAL_SECONDS` | 30 | 已优化 | 可考虑60秒（更保守） |
| `DATA_SYNC_SYMBOL_DELAY_SECONDS` | 0.5 | 合理 | 保持 |
| `BYBIT_RECV_WINDOW` | 5000 | 合理 | 保持 |
| `INTERVAL_MINUTES` | 15 | 15分钟K线 | 保持，适合趋势策略 |

### 3.4 订单处理参数

| 参数 | 当前值 | 分析 | 建议 |
|------|--------|------|------|
| 订单状态轮询超时 | 10秒 | 合理 | 保持 |
| 订单状态轮询间隔 | 0.2秒 | 可能过频繁 | 改为0.5秒，使用指数退避 |
| `STOP_ORDER_POLL_SECONDS` | 10 | 止损单轮询间隔 | 保持 |

---

## 四、代码优化建议

### 4.1 优化速率限制器配置

**文件**: `shared/exchange/bybit.py`

```python
def __init__(self, ...):
    # ... existing code ...
    
    # 优化后的预算配置（更符合Bybit实际限制）
    # 市场数据：保守配置（20个交易对）
    self.limiter.ensure_budget("market_data", 2, 5)      # 2 RPS, 5 burst
    
    # 账户查询：中等频率
    self.limiter.ensure_budget("account", 3, 6)          # 3 RPS, 6 burst
    
    # 订单操作：区分创建和查询
    self.limiter.ensure_budget("order_create", 2, 4)     # 2 RPS, 4 burst
    self.limiter.ensure_budget("order_query", 5, 10)      # 5 RPS, 10 burst
```

### 4.2 优化订单状态轮询

**文件**: `shared/exchange/bybit.py`

```python
def place_market_order(self, ...):
    # ... existing code ...
    
    # 优化后的轮询逻辑
    status = "NEW"
    filled_qty = 0.0
    avg_price: Optional[float] = None
    fee_usdt: Optional[float] = None
    pnl_usdt: Optional[float] = None
    
    # 初始延迟：订单创建后需要时间处理
    time.sleep(0.3)
    
    end = time.time() + 10.0
    last_status: Optional[OrderResult] = None
    poll_interval = 0.3  # 初始间隔
    max_interval = 1.0    # 最大间隔
    
    while time.time() < end:
        st = self.get_order_status(symbol=symbol, client_order_id=client_order_id, exchange_order_id=order_id)
        last_status = st
        status = str(st.status or status)
        filled_qty = float(st.filled_qty or 0.0)
        if st.avg_price is not None:
            avg_price = st.avg_price
        
        # ... fee extraction ...
        
        # 如果已成交或取消，立即退出
        if str(status).upper() in ("FILLED", "CANCELED", "CANCELLED", "REJECTED"):
            break
        
        # 指数退避：逐渐增加轮询间隔
        time.sleep(poll_interval)
        poll_interval = min(max_interval, poll_interval * 1.5)
```

### 4.3 添加请求头解析和动态调整

**文件**: `shared/exchange/bybit.py`

```python
def _request(self, ...):
    # ... existing code ...
    
    # 解析Bybit速率限制响应头
    if resp.status_code == 200:
        # Bybit V5可能返回的限流相关头（如果支持）
        limit_status = resp.headers.get("X-Bapi-Limit-Status")
        limit_remaining = resp.headers.get("X-Bapi-Limit-Remaining")
        limit_reset = resp.headers.get("X-Bapi-Limit-Reset-Timestamp")
        
        # 如果接近限制，记录警告
        if limit_remaining and limit_status:
            try:
                remaining = int(limit_remaining)
                if remaining < 5:  # 剩余次数少于5
                    _logger.warning(
                        f"Bybit rate limit approaching: remaining={remaining}, "
                        f"status={limit_status}, budget={budget}"
                    )
            except Exception:
                pass
    
    # ... existing error handling ...
```

### 4.4 优化closed-pnl查询

**文件**: `shared/exchange/bybit.py`

```python
def _fetch_closed_pnl(self, *, symbol: str, order_id: str, side: str) -> Tuple[Optional[float], Optional[float]]:
    """返回 (fee_usdt, pnl_usdt)。只在 SELL（平仓）时返回 pnl。"""
    if side != "SELL":
        return None, 0.0
    
    # 优化：减少查询窗口，增加查询间隔
    deadline = time.time() + 15  # 增加到15秒
    end_ms = _now_ms()
    start_ms = end_ms - 10 * 60_000  # 减少到10分钟窗口（更精确）
    
    query_count = 0
    max_queries = 5  # 最多查询5次
    
    while time.time() < deadline and query_count < max_queries:
        try:
            data = self._request(
                "GET",
                "/v5/position/closed-pnl",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "startTime": str(start_ms),
                    "endTime": str(end_ms),
                    "limit": "50",
                },
                signed=True,
                budget="account",  # 使用account预算
            )
            query_count += 1
        except ExchangeError:
            data = None
        
        # ... existing parsing ...
        
        # 增加查询间隔，避免频繁请求
        time.sleep(0.5)  # 从0.3增加到0.5秒
```

---

## 五、参数配置建议总结

### 5.1 立即调整的参数（高优先级）

```bash
# .env文件建议配置

# 1. 降低Setup B阈值，增加交易机会
SETUP_B_ADX_MIN=18              # 从20降至18
SETUP_B_VOL_RATIO_MIN=1.3       # 从1.5降至1.3
SETUP_B_AI_SCORE_MIN=50         # 从55降至50

# 2. 增加数据同步间隔（进一步减少API请求）
DATA_SYNC_LOOP_INTERVAL_SECONDS=60  # 从30增加到60秒

# 3. 根据实际账户资金设置
ACCOUNT_EQUITY_USDT=你的实际资金  # 必须设置！

# 4. 可选：增加并发持仓数（如果资金充足）
MAX_CONCURRENT_POSITIONS=5      # 从3增加到5
```

### 5.2 中期优化（需要代码修改）

1. **实现WebSocket市场数据订阅**
   - 替代REST轮询
   - 实时数据推送
   - 大幅减少API请求

2. **优化速率限制器配置**
   - 区分不同endpoint的预算
   - 动态调整基于响应头

3. **优化订单状态轮询**
   - 使用指数退避
   - 订单成交后立即退出

### 5.3 长期优化（架构改进）

1. **分离API Key**
   - 市场数据：只读API Key
   - 交易操作：交易API Key
   - 避免相互影响

2. **实现请求队列**
   - 统一管理所有API请求
   - 智能调度和优先级

3. **添加监控和告警**
   - 实时监控API请求频率
   - 接近限制时提前告警

---

## 六、实施优先级

### 阶段1：立即实施（1-2小时）
1. ✅ 调整参数配置（降低阈值，增加间隔）
2. ✅ 优化速率限制器预算配置
3. ✅ 优化订单状态轮询逻辑

### 阶段2：短期优化（1-2天）
1. 添加请求头解析和动态调整
2. 优化closed-pnl查询
3. 添加更详细的监控日志

### 阶段3：长期改进（1-2周）
1. 实现WebSocket市场数据订阅
2. 分离API Key使用
3. 实现请求队列和智能调度

---

## 七、风险提示

1. **降低阈值风险**：降低Setup B阈值会增加交易频率，需要确保风控到位
2. **增加间隔风险**：增加数据同步间隔可能导致数据延迟，影响策略判断
3. **API Key分离**：需要管理多个API Key，增加运维复杂度

---

## 八、监控建议

1. **API请求频率监控**
   - 记录每个budget的请求次数
   - 接近限制时告警

2. **错误率监控**
   - 监控retCode=10006的频率
   - 分析触发限流的原因

3. **交易执行监控**
   - 监控订单创建成功率
   - 监控订单状态查询延迟

---

## 附录：Bybit API最佳实践

1. **使用WebSocket获取市场数据**：减少REST请求90%+
2. **批量操作**：尽可能合并多个请求
3. **智能重试**：使用指数退避，避免雪崩
4. **监控响应头**：利用X-Bapi-Limit-*头信息
5. **分离API Key**：市场数据和交易操作分开
