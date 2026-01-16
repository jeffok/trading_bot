# Data-Syncer 优化方案：减少API请求频率

## 问题分析

当前问题：
- 20个交易对，每10秒轮询一次
- 每分钟 = 20交易对 × 6次 = 120次请求
- 加上gap fill和重试，实际请求可能更多
- 频繁触发Bybit速率限制（retCode=10006）

## 优化方案

### 方案1: 增加同步间隔（最简单，立即生效）

**修改**: `services/data_syncer/main.py` 第930行

```python
# 从 10秒 增加到 30-60秒
time.sleep(30)  # 或 60
```

**效果**: 
- 30秒间隔：每分钟40次请求（减少67%）
- 60秒间隔：每分钟20次请求（减少83%）

**优点**: 简单，立即生效
**缺点**: 数据更新延迟增加

### 方案2: 添加交易对间延迟（推荐）

**修改**: `services/data_syncer/main.py` 第914-929行

```python
for sym in symbols:
    try:
        sync_symbol_once(db, ex, settings, metrics, telegram, symbol=sym, instance_id=instance_id)
    except RateLimitError as e:
        sleep_s = e.retry_after_seconds or 2.0
        try:
            telegram.send(f"[RATE_LIMIT] group={e.group} sleep={sleep_s:.2f}s severe={e.severe} sym={sym}")
        except Exception:
            pass
        time.sleep(max(0.5, float(sleep_s)))
        continue
    # process a slice of precompute tasks per symbol each loop
    processed = process_precompute_tasks(db, settings, metrics, symbol=sym, max_tasks=800)
    if processed:
        logger.info(f"precompute_done symbol={sym} processed={processed}")
    
    # 添加：每个交易对之间延迟，避免突发请求
    if sym != symbols[-1]:  # 最后一个不需要延迟
        time.sleep(0.5)  # 每个交易对间隔0.5秒
```

**效果**: 
- 20个交易对 × 0.5秒 = 10秒额外延迟
- 总循环时间约20秒，每分钟约60次请求（减少50%）

### 方案3: 智能间隔调整（自适应）

根据速率限制反馈自动调整间隔：

```python
# 在main()函数中添加
last_rate_limit_ts = 0.0
base_sleep = 10.0
current_sleep = base_sleep

while True:
    # ... existing code ...
    
    for sym in symbols:
        try:
            sync_symbol_once(...)
        except RateLimitError as e:
            # 记录速率限制时间
            last_rate_limit_ts = time.time()
            current_sleep = min(60.0, current_sleep * 1.5)  # 增加间隔
            # ... existing error handling ...
    
    # 如果没有速率限制，逐渐减少间隔
    if time.time() - last_rate_limit_ts > 300:  # 5分钟无限制
        current_sleep = max(base_sleep, current_sleep * 0.9)
    
    time.sleep(current_sleep)
```

### 方案4: 使用WebSocket（长期方案，最有效）

实现WebSocket订阅实时K线数据，完全避免REST轮询。

**优点**:
- 实时数据推送，无需轮询
- 大幅减少API请求
- 更及时的数据更新

**实现步骤**:
1. 添加WebSocket客户端支持
2. 订阅所有交易对的K线流
3. 收到数据后直接写入数据库
4. 保留REST作为兜底（初始化和补漏）

## 推荐实施顺序

1. **立即实施**: 方案1 + 方案2（5分钟完成）
   - 增加基础间隔到30秒
   - 添加交易对间延迟0.5秒
   - 预期减少70%+的请求

2. **短期优化**: 方案3（1-2小时）
   - 实现自适应间隔
   - 根据速率限制反馈调整

3. **长期优化**: 方案4（1-2天）
   - 实现WebSocket支持
   - 完全替代REST轮询

## 当前代码状态

✅ **已确认**: `fetch_klines` 使用公开API（`signed=False`）
✅ **已确认**: 速率限制错误已正确处理（抛出RateLimitError）
⚠️ **需要优化**: 请求频率过高

## Bybit速率限制参考

- HTTP REST: 每个IP每5秒最多600次请求
- WebSocket: 每个IP 5分钟内最多500个连接
- 建议: 使用WebSocket订阅公共数据流
