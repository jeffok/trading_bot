# 修改日志（CHANGELOG）

## 2026-01-09 - Iteration 1

### 新增
- 新增 `control_commands` 领域模块（`shared/domain/control_commands.py`），支持写入/拉取/标记控制指令。
- API-Service：/admin 系列接口在写 `system_config` 的同时写入 `control_commands`，形成可消费的审计队列。
- API-Service：增强 `/health` 输出（包含各服务 last_seen、行情滞后、HALT/EMERGENCY 标志）。
- Data-Syncer：指标计算新增 Keltner Channel（20）/ Squeeze 状态 / RSI 斜率 / BTC 相关性（滚动 96，best-effort）。
- Strategy-Engine：实现 Setup B（V8.3）判定逻辑（Squeeze 释放 + 动量转正 + 量能放大 + ADX/+DI/-DI 趋势确认 + AI 分数门槛）。

### 优化
- Strategy-Engine：新增风险预算硬约束（默认 3% equity），超预算自动降杠杆；仍超预算则拒绝开仓并落库事件。
- Strategy-Engine：新增简易 Circuit Breaker（速率限制/失败次数阈值触发 HALT_TRADING）。
- Admin-CLI：halt/resume/emergency-exit/set 同步写入 `control_commands` 以便 Strategy-Engine 消费。

### 说明
- BTC 相关性为 best-effort：若 BTC 行情缺失或对齐不足则输出 `null`。
- Circuit Breaker 仅做轻量级保护：优先保证主循环稳定，不追求复杂策略。

## 2026-01-09 Iteration 2
- api-service: 增强 /health 输出（服务心跳快照、HK/UTC 时间、HALT/EMERGENCY 标记、指定 symbol 的 market_data_lag）。
- api-service: 在生命周期中增加心跳线程（按 HEARTBEAT_INTERVAL_SECONDS 写入 service_status），便于 /health 与 /admin/status 观测。
- 无破坏性变更：保留 /admin/status 原有输出。
## 2026-01-09 (iter3)

- order_events：新增 raw_payload_json 字段（迁移 0008），写入前对 payload 进行递归脱敏（token/secret/signature 等字段替换为 ***），并截断超长字符串，满足 V8.3 “raw_payload_json 必须脱敏”的要求。
- strategy-engine：修正 Setup B reason_code 类型为枚举（ReasonCode），并将 BUY 成交事件的 reason_code/reason 与 Setup B 决策保持一致（不再使用通用 'Order placed'）。
- strategy-engine：AI 选币阶段改用 last_two_cache + setup_b_decision（包含 prev bar + ai_score），并将 open_reason_code/open_reason 写入候选 meta，保证选币阶段与实际开仓阶段判定一致。
- strategy-engine：开仓成交 Telegram 告警补充 reason_code/reason 字段，满足“理由必须出现在告警”要求。
- Telegram：send_alert_zh 自动注入 ts_hk / ts_utc（HK/UTC 时间戳），满足告警可追溯要求。


## 2026-01-09 - Iteration 4

### 改进
- Telegram「开仓成交 / BUY_FILLED」告警补齐关键字段：ai_score、stop_price、stop_dist_pct、reason_code/reason、client_order_id、exchange_order_id、保护止损单号（stop_client_order_id / stop_exchange_order_id）。
- 新增轻量结构化日志 helper：`shared/telemetry/action_log.py`（log_action），用于输出 action/reason_code/reason/trace_id 等字段为一行 JSON，便于检索与审计。
- 在风控拒单/降杠杆与熔断 HALT 路径补充 `log_action` 结构化日志，确保异常路径同样可追溯。

### 新增
- 新增自检脚本 `tools/self_check.py`：执行 compileall + 校验 order_events 脱敏逻辑（无需额外依赖）。

### 已知限制
- Telegram 其他事件（平仓/止损/拒单等）字段仍可能不完全一致；本轮优先补齐最关键的开仓成交与风控/熔断路径，下一阶段统一封装 trade alert formatter。

## 2026-01-09 - Iteration 5

### 目标
- 统一告警/日志格式（trade 类 Telegram 告警字段口径统一；关键动作日志结构化）。

### 新增
- 新增 `shared/telemetry/trade_alerts.py`：
  - `build_trade_summary()`：统一 trade 告警 summary_kv 字段口径（event/trace_id/exchange/symbol/side/qty/price/leverage/ai_score/stop_price/stop_dist_pct/reason_code/reason/client_order_id 等）
  - `send_trade_alert()`：统一发送入口，避免各处字段缺失/不一致。

### 改进
- Strategy-Engine：将以下告警改为使用统一 trade formatter（字段口径一致）并补充结构化日志 `log_action`：
  - 开仓成交（BUY_FILLED）
  - 平仓成交（SELL_FILLED）
  - 触发止损（STOP_LOSS）
  - 交易所保护止损成交（PROTECTIVE_STOP_FILLED）
  - 风控拒单（RISK_REJECT）
  - 保护止损挂单失败降级（STOP_ARM_FAILED_FALLBACK）
- Telegram：`send_alert_zh` 输出顺序统一（优先输出 ts_hk/ts_utc/event/trace_id/exchange/symbol/side/qty/price/...，剩余字段按 key 排序），提升可读性与一致性。
- API-Service：管理类告警发送后补充 `log_action` 结构化日志（含 event/trace_id/level 等），便于集中检索。
- tools/self_check.py：改为仅做 compileall 自检（不依赖数据库驱动），避免环境缺少 pymysql 时无法运行。

### 已知限制
- 仍有少量非 trade 类告警（如系统启停/紧急退出）未迁移到统一 formatter，但 Telegram 输出顺序已统一，且关键字段仍可追溯。

## Iteration 6 - 2026-01-09
### 目标
- 最终收尾：统一系统类告警/日志格式；修复遗留的“截断/省略号”代码；提供可选的 SGDClassifier 口径实现；补齐最小回归自检。

### 完成
- 统一系统类告警 formatter：
  - 新增 `shared/telemetry/system_alerts.py`：`build_system_summary()` / `send_system_alert()`
  - `api-service` 的 `tg_alert()` 改为基于 `build_system_summary()` + `send_system_alert()`，并同步写 `log_action()`（JSON）
  - `strategy-engine` 的 control_commands（HALT/RESUME/EMERGENCY_EXIT）与熔断告警全部迁移到 `send_system_alert()`
  - `tools/admin_cli` 的 set/halt/resume/emergency/smoke/e2e 告警全部迁移到 `send_system_alert()`，并补充 `log_action()`

- 修复遗留的“省略号/截断行”：
  - 修复 `tools/admin_cli/__main__.py` 中 `write_control_command(...)` 与告警 payload 的截断行
  - 修复 `strategy-engine` 中 RateLimitError 分支的重复 try/截断行，确保可编译可运行

- AI：提供可选的 SGDClassifier 口径实现（不依赖 sklearn）
  - 新增 `shared/ai/sgd_compat.py`：`SGDClassifierCompat`（支持 `partial_fit/predict_proba/to_dict/from_dict`）
  - Settings 新增 `AI_MODEL_IMPL=online_lr|sgd_compat`（默认 online_lr）
  - 策略引擎加载模型时会根据存储的 `impl` 或 `AI_MODEL_IMPL` 选择实现

### 已知限制
- `SGDClassifierCompat` 为轻量实现（不依赖 sklearn），若你验收必须“sklearn.SGDClassifier”，需要在运行镜像中引入 sklearn 并替换实现。
- 一些极少数非关键异常路径仍可能只写日志不发告警（例如网络抖动导致的瞬时错误），但关键交易/风控/熔断/控制路径已统一。

## Iteration 8 - 2026-01-09

### 目标
- 一次性收敛“文档必需项”缺口：时区、/health 必备字段、control_commands 秒级生效、AI 模型 ai_models(is_current) 持久化、全链路 ERROR 落库审计、管理查询接口。

### 完成
- Docker/Compose：所有服务设置 `TZ=Asia/Hong_Kong`，镜像安装 `tzdata`，确保容器内 IANA 时区可用。
- API-Service：
  - 新增全局异常 handler：未捕获异常会写入 `order_events(ERROR)`（带 trace_id/path/method）。
  - /health：补齐 `engine_last_tick`（包含 last_tick_ts_utc/hk）与 `recent_errors`（最近 10 条 ERROR 摘要）。
  - 新增管理查询接口：
    - `GET /admin/control_commands`：查看控制命令队列（NEW/PROCESSED/ERROR/ALL）。
    - `GET /admin/ai_models`：查看 AI 模型元数据（含 current 标记与 blob 大小）。
- Data-Syncer：主同步循环异常写入 `order_events(ERROR)`，提升可观测性与审计一致性。
- Strategy-Engine：
  - control_commands 轮询线程：每 `CONTROL_POLL_SECONDS`（默认 2s）消费 NEW 命令，满足 1-3 秒生效要求。
  - AI：优先从 `ai_models`（is_current=1）加载模型；训练后同时持久化到 `ai_models` 并保留 system_config 兼容写入。
  - /health 需要字段：在 `service_status` 状态快照写入 `last_tick_ts_utc/last_tick_ts_hk`。
  - RateLimit：`RATE_LIMIT_BACKOFF` 统一使用 `ReasonCode.RATE_LIMIT_429`。

### 安全/交付
- 打包发布时移除根目录 `.env`（避免泄漏），保留 `.env.example`。



## Iteration 9 - 2026-01-11

### 目标
- 完成“仍未完全完成/与文档不一致”清单（V8.3 对齐）：feature_version、统一 client_order_id 口径、交易锁 key、tick budget、5 分钟 position snapshot、control_commands 审计字段。

### 完成
- Feature Cache Versioning（feature_version）
  - 新增 `FEATURE_VERSION`（默认 1），并在 `market_data_cache / market_data_cache_history / precompute_tasks` 增加 `feature_version` 列与主键维度（migration: `0010_feature_version_support.sql`）。
  - `data-syncer`：precompute_tasks/market_data_cache 写入与查询全部带 `feature_version`。
  - `strategy-engine`：读取缓存（latest_cache/last_two_cache）加入 `feature_version` 过滤。
  - `api-service` 与 `tools/admin_cli` 的 market_data_cache 查询加入 `interval_minutes + feature_version` 维度，避免混用不同版本缓存。

- client_order_id（V8.3 口径）
  - `shared/domain/idempotency.py`：实现 `asv8-{symbol}-{side}-{timeframe}-{bar_close_ts}-{nonce}`，nonce 从 trace_id 派生稳定短 hash，保证重试幂等。
  - `strategy-engine`：所有下单/平仓/止损相关 client_order_id 生成调用均补齐 `interval_minutes + trace_id`。

- 分布式交易锁（V8.3 口径）
  - `strategy-engine`：锁 key 统一为 `asv8:lock:trade:{symbol}`，并新增 `TRADE_LOCK_TTL_SECONDS`（默认 30s）。

- Tick Budget（V8.3）
  - Settings 新增 `TICK_BUDGET_SECONDS`（默认 10s）。
  - `strategy-engine`：tick 内加入 budget 检查，超时会提前结束当次 tick 并写统一日志（ReasonCode: `TICK_TIMEOUT`）。

- Position Snapshots（每 5 分钟）
  - Settings 新增 `POSITION_SNAPSHOT_INTERVAL_SECONDS`（默认 300s）。
  - `strategy-engine`：在 tick 间隔 sleep loop 中，每 N 秒为所有有持仓（base_qty>0）的 symbol 写一条 snapshot（meta.note=periodic_snapshot）。

- control_commands 审计字段（字段化）
  - 新增 migration: `0011_control_commands_audit_columns.sql`：control_commands 增加 `trace_id/actor/reason_code/reason`。
  - `shared/domain/control_commands.py`：写入/读取支持新字段。
  - `api-service`：写入控制命令时同步写入上述字段；`GET /admin/control_commands` 返回新字段。

### 已知限制
- `0010_feature_version_support.sql` 会重建主键（DROP/ADD PRIMARY KEY），首次升级会触发表锁与索引重建，请在低峰期执行。


## Iteration 10 - 2026-01-11

### 目标
- 补齐文档“推荐但可能用于验收”的两项：
  1) `order_events` 增加 `action/actor/event_ts_hk` 字段化审计能力。
  2) data-syncer 数据延迟（lag）超过阈值时发送 Telegram 告警（并带冷却时间防刷屏）。

### 完成
- order_events 审计字段化
  - 新增 migration: `0012_order_events_audit_columns.sql`
    - `order_events` 增加 `action`、`actor`、`event_ts_hk`（DATETIME）。
    - 增加索引：`(trace_id, created_at)`、`(action, created_at)`。
  - `shared/domain/events.py`
    - `append_order_event()` 新增可选参数 `action/actor`（默认：action=event_type, actor=service）。
    - 写入时计算 `event_ts_hk`（UTC+8，naive DATETIME），满足“HK 时间审计字段”需求。

- data-syncer lag 阈值 Telegram 告警
  - Settings 新增：
    - `MARKET_DATA_LAG_ALERT_SECONDS`（默认 120 秒）
    - `MARKET_DATA_LAG_ALERT_COOLDOWN_SECONDS`（默认 300 秒）
  - `data-syncer`：在计算 `market_data_cache` lag 时，若 `now - bar_close_time` 超阈值，发送系统类告警 `DATA_LAG`（含 symbol/interval/feature_version/lag/threshold/last_open/last_close）。
  - 内存级 per-symbol 冷却（cooldown）避免刷屏。

### 配置示例
- 见 `.env.example`：已新增 lag 告警相关环境变量。

## Iteration 10 Hotfix - 2026-01-11

### 修复
- 修复 migrations runner 解析 `.sql` 时对注释中分号不兼容的问题：
  - 迁移文件的注释行可能包含 `;`，原逻辑按 `;` 分割会把注释切碎并当作 SQL 执行，触发 MariaDB 1064。
  - 现在在 split 前会先清理 `-- ...` 与 `/* ... */` 注释（轻量实现），再执行 SQL 语句列表。

## Iteration 10 Hotfix2 - 2026-01-11

### 修复
- 修复 `get_instance_id()` 调用不一致导致 `api-service` 启动时报错：
  - 兼容 `get_instance_id(default)` 与 `get_instance_id(service, default)` 两种调用方式。
  - 避免 `TypeError: get_instance_id() takes from 0 to 1 positional arguments but 2 were given`。

## Iteration 10 Hotfix3 - 2026-01-11

### 修复
- 修复 `data-syncer` 查询 `market_data` 时传参数量与 SQL 占位符不匹配导致循环报错：
  - 现象：`sync_error ... err=not all arguments converted during string formatting`
  - 原因：`market_data` 表不包含 `feature_version` 维度，但代码在 `WHERE symbol=%s AND interval_minutes=%s` 的查询中额外传入 `feature_version` 参数。
  - 修复：移除多余参数，仅传 `(symbol, interval)`。
