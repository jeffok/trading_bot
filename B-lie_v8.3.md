# Alpha-Sniper-V8 量化交易系统需求与技术规格说明书（B-lite 企业级实施指南）
文档版本：V8.3（开发落地详版）  
日期：2026-01-06（Asia/Hong_Kong）  
项目代号：Alpha-Sniper-V8  
目标读者：Python 开发工程师 / 量化工程师 / DevOps  

---

## 变更记录
- V8.2：最终实施指南版（你提供的基础稿，含香港时间、动态限流、日志与告警示例、CLI 使用示例等）
- V8.3：在 V8.2 基础上**补齐“怎么做”的细节**：  
  1) 各服务的输入输出与状态机、幂等点、异常处理与恢复策略  
  2) 参数更新小工具（Admin CLI）的完整职责、命令与审计要求  
  3) “交易所 API 智能节流”的工程级实现规范（按响应动态调参）  
  4) Telegram 告警与日志：必须覆盖**下单/平仓/止损**，并强制携带“理由”与可追溯字段  
  5) 数据库表与字段的“可落地”约束（唯一键、枚举、字段含义、建议迁移）

---

# 0. 交付目标（必须达成）
> 本系统是 500 USDT 本金的 AI 增强趋势交易系统，通过 15 分钟 K 线自动交易主流币种，工程重点是稳定性、可审计、可控与自我保护（智能节流与风控）。

## 0.1 最小可验收交付（MVP）
1) 三服务（strategy-engine / data-syncer / api-service）可通过 docker-compose 启动并运行  
2) 外部依赖：你已有 PostgreSQL 与 Redis，本项目不负责安装，只负责连接、建表/升级表  
3) 支持：停开仓、恢复、紧急清仓、参数热更新（API + CLI 两条路径）  
4) 订单事件流 order_events 记录完整：下单 → 提交 → 成交/撤单/拒单/异常  
5) 必须有：智能节流（按交易所返回的 rate limit 动态调整）  
6) 必须有：Telegram 告警与结构化日志，覆盖下单/平仓/止损，且包含理由与追踪字段  
7) 重启恢复：任一服务重启后，不重复下单、不重复写入同一根 K 线、不重复归档同一范围  

## 0.2 关键工程指标（SLO）
- strategy-engine：每个 15m tick 的决策与执行，在 10 秒内完成（超时应降级或跳过本 tick）
- data-syncer：15m K 线延迟（数据落库时间 - K 线开盘时间）< 2 分钟（可配置告警阈值）
- api-service：/health < 200ms，/admin 操作 < 1s（DB 正常时）

---

# 1. 部署形态与服务边界（B-lite）

## 1.1 总体架构（文本图）
- data-syncer：专门拉数据 + 计算指标 + 写缓存 + 归档  
- strategy-engine：核心交易决策与执行 + 审计写入  
- api-service：管理控制面 + 健康检查 + 指标输出 + 告警编排  

## 1.2 运行时与时区规范（强制）
- 运行时调度时区：**香港时间（Asia/Hong_Kong）**  
- 数据库存储时间：**UTC**（所有 *_utc 字段一律 UTC）
- Docker / 进程：必须设置 `TZ=Asia/Hong_Kong`  
- 任何基于“每天凌晨”的任务（归档等）以香港时间触发，落库使用 UTC

---

# 2. 全局不变量（所有开发人员必须遵守）

## 2.1 幂等不变量
- **client_order_id 是系统级幂等键**：相同交易机会的所有重试必须复用同一个 client_order_id  
- 任何下单动作必须满足：  
  1) 先写 `order_events(CREATED)`  
  2) 再调用交易所 API  
  3) 成功/失败都必须写事件（SUBMITTED/FILLED/ERROR/REJECTED/…）

## 2.2 不可变事件流
- `order_events`：只允许 INSERT，不允许 UPDATE/DELETE（保留全部历史）

## 2.3 “理由”强制
- 任何触发交易或控制动作，必须提供：  
  - `reason_code`（短码，可检索）  
  - `reason`（可读解释，1-2 句话）  
- 理由必须同时出现在：  
  1) 结构化日志（action/reason_code/reason）  
  2) Telegram 告警（同样字段）  
  3) DB 审计字段（order_events.note 或专用字段，见 3.2）

## 2.4 安全不变量
- API Key/Secret、Admin Token、DB 密码：不得写入日志/告警/DB  
- `order_events.raw_payload_json` 必须脱敏（移除敏感字段）

---

# 3. 数据库设计（可实施版）

## 3.1 表清单（必须实现）
- 服务与控制：
  - `schema_migrations`
  - `service_status`
  - `system_config`
  - `config_audit`
  - `control_commands`
- 交易审计：
  - `order_events`
  - `trade_logs`
  - `position_snapshots`
- 行情与缓存：
  - `market_data`
  - `market_data_cache`
  - `archive_audit`
  - `*_history` 系列表（归档目标）
- AI：
  - `ai_models`

## 3.2 推荐新增字段（V8.3 强烈建议）
如果 migrations 尚未固定，建议通过新增迁移脚本加入以下字段，以减少“把 JSON 塞 note”的混乱。

### 3.2.1 order_events（建议新增）
- `trace_id`：VARCHAR(64)  
- `action`：VARCHAR(32)（与日志 action 对齐）  
- `reason_code`：VARCHAR(64)  
- `reason`：TEXT  
- `actor`：VARCHAR(64)（system/admin:<name>）  
- `event_ts_hk`：DATETIME（可选，用于方便查询；真实时间仍以 event_ts_utc 为准）

索引建议：
- idx_order_events_symbol_ts： (symbol, event_ts_utc)
- idx_order_events_client_order： (client_order_id)
- idx_order_events_trace： (trace_id)

### 3.2.2 control_commands（建议新增）
- `trace_id`、`actor`、`reason_code`、`reason`（或统一放到 payload_json，但要规范 schema）

### 3.2.3 trade_logs（建议新增）
- `close_reason_code` / `close_reason`（明确平仓/止损原因）
- `stop_price` / `stop_dist_pct`

## 3.3 关键唯一键（必须）
- `market_data`：UNIQUE(symbol, timeframe, kline_open_ts_utc)
- `market_data_cache`：UNIQUE(symbol, timeframe, kline_open_ts_utc, feature_version)
- `order_events`：建议 UNIQUE(client_order_id, event_type)（或加 event_ts_utc）避免重复写同一状态  
- `service_status`：PRIMARY(service_name)（每服务一行，更新心跳使用 UPSERT）

## 3.4 归档幂等（必须）
- `archive_audit` 记录每次归档范围（table、cutoff、rows、status、trace_id）
- 归档移动使用分批事务，失败可重试，不重复移动（靠 history 表唯一键 + audit 状态）

---

# 4. 服务规格（做什么、怎么做、做到什么程度）

# 4.1 strategy-engine（策略引擎）

## 4.1.1 核心循环（15m tick）
### 触发规则（香港时间）
- tick 判定：分钟 % 15 == 0 且秒数接近 0（允许误差窗口，例如 0-3 秒）

### 运行步骤（必须）
1) 读取全局开关与配置（system_config）
2) 检查 HALT 状态（来自 control_commands 或 system_config）  
3) 对每个 symbol：
   - 获取分布式锁（Redis）`asv8:lock:trade:{symbol}`（防止多实例重复下单）
   - 读取 `market_data_cache`（优先），缺失则使用兜底计算（可选）
   - 生成信号（见 6）
   - AI 评分（见 7）
   - 风控计算与校验（见 8）
   - 生成 `client_order_id`
   - 写 `order_events(CREATED)`（含 reason）
   - 执行下单（走智能节流入口，见 5）
   - 写 `order_events(SUBMITTED)` 或 `order_events(ERROR/REJECTED)`
   - 成交确认：
     - 最小实现：轮询订单状态直到 FILLED 或超时
     - 成功：写 `order_events(FILLED)` + 写 `position_snapshots`
   - **开仓后必须处理止损**（见 6.4）
4) 每 5 分钟写一次 `position_snapshots`（可用独立定时器）
5) 处理 control_commands（至少每 1-3 秒轮询一次）
6) 写 `service_status` 心跳（每 5-10 秒）

## 4.1.2 幂等实现点（必须）
- client_order_id 生成规则：  
  `asv8-{symbol}-{side}-{timeframe}-{bar_close_ts}-{nonce}`  
  - bar_close_ts 用 UTC 毫秒（由香港时间推导）
  - nonce 用递增序号或随机短串
- 重试策略：  
  - 网络失败重试必须复用 client_order_id  
  - 写事件时必须防重（用 UNIQUE 或先查后写）

## 4.1.3 状态机（必须对齐）
事件类型建议：
- CREATED / SUBMITTED / ACK（可选） / PARTIAL（可选） / FILLED / CANCELED / REJECTED / ERROR / RECONCILED  
要求：
- 任意异常必须落 ERROR 事件（含 reason_code）

---

# 4.2 data-syncer（数据同步与预计算）

## 4.2.1 同步范围（必须）
- symbol 列表来自 system_config（例如 `symbols=BTCUSDT,ETHUSDT`）
- timeframe 至少支持 15m

## 4.2.2 同步策略（必须）
1) 增量拉取：从 DB 查询每个 symbol 的最新 kline_open_ts_utc，然后向交易所拉取后续数据  
2) 缺口检测：若连续 K 线不连续（> 15m），记录 gap 并执行补洞  
3) 清洗：排序、去重（依赖唯一键）、异常标记（可选）  
4) 指标预计算：ADX/DI、EMA21/55、Squeeze、Momentum、VolRatio、RSI_slope → 写入 market_data_cache  
5) 心跳：写 service_status  
6) 归档：每日凌晨（香港时间）执行 cutoff=now_utc-90d 的迁移，写 archive_audit  

---

# 4.3 api-service（控制面）

## 4.3.1 必须接口
- GET /health  
- GET /metrics  
- POST /admin/halt  
- POST /admin/resume  
- POST /admin/emergency_exit  
- POST /admin/update_config  
- GET /admin/status  

## 4.3.2 /admin 写接口统一要求（必须）
- Body 必须包含：
  - actor（操作者）
  - reason_code
  - reason
- 行为必须：
  - 生成 trace_id
  - 写 control_commands 或 system_config/config_audit
  - 写结构化日志 + Telegram（按类型）
  - 返回 ok + trace_id

## 4.3.3 安全（必须）
- /admin/* 必须 Bearer Token  
- 建议限流（Redis）

---

# 4.4 参数更新小工具（Admin CLI）（必须交付）

## 4.4.1 CLI 入口
- 推荐模块：`python -m admin_cli ...`
- 推荐运行位置：api-service 容器内

## 4.4.2 必须命令
- status
- halt/resume/emergency-exit（均要求 --by/--reason_code/--reason）
- set/get/list（set 要求 --by/--reason_code/--reason）

## 4.4.3 审计一致性（必须）
CLI 写操作必须同时写：
- control_commands 或 config_audit/system_config
- 结构化日志（action + reason）
- Telegram（关键动作）

---

# 5. 交易所接入与“智能节流”（必须实现）

## 5.1 接入分层（必须）
- exchange_client（签名、请求、解析响应）
- adaptive_rate_limiter（节流/退避/动态调参）
- exchange_gateway（统一业务接口）

## 5.2 分组限流（必须）
- market_data / account / order 三组独立预算

## 5.3 动态调参（必须）
- 解析响应头（如 used-weight/order-count）
- 429/418：
  - 退避（指数退避 + 抖动）
  - 若 Retry-After 存在优先使用
  - 写日志 action=RATE_LIMIT_BACKOFF，reason_code=RATE_LIMIT_429
- 持续限流超阈值：
  - strategy-engine 自动 HALT + Telegram（含建议）

## 5.4 统一入口约束（必须）
- 禁止绕过 limiter 直接调用 HTTP
- limiter 必须暴露 metrics：requests_total、wait_seconds、429_total、backoff_seconds

---

# 6. 策略与止损（必须可解释）

## 6.1 指标来源
- 优先：market_data_cache

## 6.2 Setup B（先落地）
触发条件（做多示例）：
- ADX>阈值 且 +DI>-DI
- Squeeze 释放
- 动量由负转正
- VolRatio>阈值
- AI score >= 阈值

理由输出（必须）：
- reason_code：SETUP_B_SQUEEZE_RELEASE
- reason：Squeeze 释放 + 动量转正 + 量能放大，ADX 强趋势确认

## 6.3 Setup A（后续）
- EMA 回踩 + 形态识别（逐步增强）

## 6.4 止损（必须）
- 开仓后必须设置 stop_price 与 stop_dist_pct
- 实盘：下发止损单（如 STOP_MARKET）
- 纸交易：触发条件模拟平仓
- 止损触发必须写：
  - order_events（止损触发/止损成交）
  - trade_logs.close_reason_code=STOP_LOSS
  - Telegram：🔴 止损成交（含原因）
  - 日志：action=STOP_LOSS（含原因）

---

# 7. AI（SGDClassifier 增量学习）

## 7.1 冷启动（必须）
- 默认 ai_score=50
- 样本不足时禁用仓位放大（或更严格风控）

## 7.2 训练触发（必须）
- 平仓写 trade_logs 后触发 partial_fit
- 模型落库 ai_models，维护 is_current

---

# 8. 风控（必须可追溯）

## 8.1 动态保证金
- base_margin=max(50,equity*10%)
- ai_score>85：base_margin*=1.2

## 8.2 风险预算（3%硬约束）
- risk_amount = base_margin * leverage * stop_dist_pct
- 不满足则降杠杆，仍不满足则拒单并写理由（日志+事件）

## 8.3 熔断（必须）
- 连续失败/持续限流/回撤阈值触发 HALT + Telegram

---

# 9. 可观测性与一致性（日志=告警=审计）

## 9.1 /health 必须包含
- 三服务状态与心跳
- data-sync lag
- engine halt 状态与最近 tick
- 最近错误摘要

## 9.2 /metrics 必须包含
- heartbeats
- orders_total/latency
- data_sync_lag/gap
- rate_limit 指标
- telegram_send_total

## 9.3 Telegram（必须覆盖下单/平仓/止损且含理由）
- 开仓提交/成交
- 平仓提交/成交
- 止损触发/成交
每条必须包含：时间(HK+UTC)、symbol、价格、数量、杠杆、止损、AI、风控、trace_id、client_order_id、reason_code/reason

## 9.4 结构化日志（必须）
每条交易动作日志必须包含：action + reason_code + reason + trace_id + client_order_id（如有）

---

# 10. 自测与验收（开发人员对照）
- /admin 与 CLI 写操作强制 reason_code/reason/actor
- 智能节流：解析响应头 + 429 退避 + metrics
- 幂等：重启不重复下单
- 归档：失败可重试不重复
- 止损：触发后事件/日志/告警/交易日志一致

（文档结束）
---

# 11. 阶段开发建议（Roadmap）

> 目标：以“可上线、可审计、可控、可恢复”为第一优先级；先把工程骨架与关键不变量做对，再逐步增强策略与 AI。

## Phase 0：工程骨架与本地可跑（0.5-1 周）
- 代码仓库结构与基础脚手架（3 服务 + 公共库）
- docker-compose 一键启动（PostgreSQL/Redis 作为外部依赖仅做连接配置）
- 基础配置系统（env + system_config 读取）与时区规范（TZ=Asia/Hong_Kong，DB 用 UTC）
- 基础日志框架（结构化日志）与 trace_id 贯穿

交付验收：
- compose 启动 3 服务，/health 返回 OK（包含版本、时区、DB/Redis 连接状态）

## Phase 1：MVP 闭环（1-2 周）
重点是“能稳定交易且不重复下单”：
- data-syncer：15m K 线增量同步 + UNIQUE 去重 + 缺口检测（gap 记录即可）
- market_data_cache：指标预计算最小集（至少能支撑 Setup B）
- strategy-engine：15m tick 调度 + 分布式锁 + client_order_id 幂等 + order_events 不可变事件流
- exchange 接入：exchange_client + exchange_gateway + adaptive_rate_limiter（429/418 退避 + 解析响应头）
- api-service：/admin halt/resume/emergency_exit/update_config + Bearer Token
- Telegram：覆盖下单/平仓/止损（含 reason_code/reason/trace_id）

交付验收（对齐 MVP 要求）：
- 订单事件从 CREATED → SUBMITTED → FILLED/ERROR 全链路完整
- 重启任一服务不重复：不重复下单、不重复写同一根 K 线、不重复归档同一范围
- 智能节流生效（具备 429 退避与指标）

## Phase 2：可观测性 + 恢复能力（1-2 周）
- /metrics（Prometheus）完善：orders_total/latency、rate_limit、data_sync_lag/gap、telegram_send_total
- service_status 心跳、最近错误摘要、data-sync lag 监控与告警
- 订单对账（RECONCILED）最小实现：启动时加载未终态订单并拉取状态补齐
- 归档：archive_audit + history 表迁移（分批事务 + 幂等）
- Admin CLI 全量落地：status/halt/resume/emergency-exit/set/get/list，写审计与告警一致

交付验收：
- 能定位“为什么没下单/为什么被风控拒单/为什么被限流”，并能通过 trace_id 回溯
- 故障场景演练：DB 短暂不可用、交易所 429、服务重启、网络抖动

## Phase 3：策略与 AI 增强（持续迭代）
- Setup A（EMA 回踩 + 形态）逐步上线（灰度：只打分不交易 → 小仓位 → 全量）
- AI：SGDClassifier partial_fit 完整流水（trade_logs 写入触发训练，ai_models 版本管理）
- 风控增强：回撤熔断、连续失败熔断、动态仓位（基于 ai_score）
- 回测/仿真：统一回放接口，复用 strategy-engine 信号与风控模块

## Phase 4：生产化（可选）
- 多实例（HA）：分布式锁 + 领导者选举（可选）
- 灾备：DB 备份策略、migrations 管理、密钥轮换
- 安全：更严格的 admin 操作审计与 IP 白名单/二次确认（按需要）
---

# 12. 项目目录与文件命名（建议）

> 目标：三服务共享同一套“领域模型 + 事件/审计 + 交易所网关 + 风控/策略基础库”，减少重复与分叉。

## 12.1 顶层目录（建议）
```text
alpha-sniper-v8/
  README.md
  docker-compose.yml
  .env.example
  pyproject.toml
  poetry.lock (可选)
  Makefile (可选)
  scripts/
    init_db.sql (可选)
    wait_for_db.sh
  migrations/
    0001_init.sql
    0002_add_reason_fields.sql
    0003_archive_history_tables.sql
  shared/
    __init__.py
    config/
      __init__.py
      loader.py
      defaults.py
      schema.py
    logging/
      __init__.py
      logger.py
      trace.py
      sanitize.py
    db/
      __init__.py
      maria.py
      migrations.py
      models.py
      repo.py
    redis/
      __init__.py
      client.py
      locks.py
      rate_limit_store.py
    exchange/
      __init__.py
      client.py
      gateway.py
      rate_limiter.py
      errors.py
    domain/
      __init__.py
      enums.py
      ids.py
      time.py
      events.py
      risk.py
      strategy.py
      ai.py
    telemetry/
      __init__.py
      metrics.py
      telegram.py
  services/
    data_syncer/
      Dockerfile
      __init__.py
      main.py
      scheduler.py
      syncer.py
      indicators.py
      archival.py
      health.py
    strategy_engine/
      Dockerfile
      __init__.py
      main.py
      tick.py
      signal.py
      executor.py
      reconciler.py
      positions.py
      health.py
    api_service/
      Dockerfile
      __init__.py
      main.py
      routes/
        __init__.py
        health.py
        metrics.py
        admin.py
      auth.py
      health.py
  tools/
    admin_cli/
      __init__.py
      __main__.py
      commands/
        __init__.py
        status.py
        halt.py
        resume.py
        emergency_exit.py
        config_get.py
        config_set.py
        config_list.py
  tests/
    unit/
    integration/
```

## 12.2 文件命名与职责映射
- `shared/exchange/rate_limiter.py`：adaptive_rate_limiter 的核心实现（分组预算 + 429/418 退避 + metrics）  
- `shared/domain/events.py`：order_events 事件类型、写入约束、reason 字段模型  
- `services/strategy_engine/reconciler.py`：启动/定时对账未终态订单，补写 RECONCILED（最小实现）  
- `services/data_syncer/archival.py`：archive_audit + history 表迁移（分批事务、可重试幂等）  
- `scripts/trading_test_tool/*`：所有写操作都必须带 by/reason_code/reason，并写审计/日志/Telegram（与 /admin 对齐）。⚠️ 仅在Docker中使用。

## 12.3 DB 迁移文件（建议命名）
- `0001_init.sql`：建表（schema_migrations、service_status、system_config、config_audit、control_commands、order_events、trade_logs、position_snapshots、market_data、market_data_cache、archive_audit、ai_models 等）  
- `0002_add_reason_fields.sql`：补齐 trace_id/action/reason_code/reason/actor 等字段（如需）  
- `0003_archive_history_tables.sql`：创建 *_history 表及必要 UNIQUE/索引

---

## 13. 项目需求与架构（V8.0 版本历史）

> **注**: 以下内容来自 V8.0 版本需求文档，已整合到本文档。保留作为历史参考。

### 13.1 核心目标

利用 500 USDT 本金，通过高胜率趋势交易与 AI 优选，实现周盈利 100U - 200U (20% - 40% ROI)。

### 13.2 系统架构与运行时

本版本采用微服务架构思想，确保高可用性、可观测性与自愈能力。

**容器化进程管理**:
- **基础镜像:** Python 3.11 (Slim)
- **进程编排:** 三个独立服务（data-syncer / strategy-engine / api-service）
- **自愈机制**: Docker Healthcheck + 自动重启

**监控与告警**:
- **Healthcheck:** 暴露 `GET /health` 端点，返回各服务状态
- **Alerting:** 关键事件通过 Telegram Bot 发送告警

### 13.3 数据库设计：高性能与审计

采用 **PostgreSQL**，数据以JSONB格式存储，针对高频读写、审计溯源进行深度优化。

**高性能存储**:
- **分区表**: `trade_logs` 按 `timestamp` 进行 RANGE 分区（按月）
- **预计算缓存**: `market_data_cache` 存储预计算好的 ADX, Squeeze Momentum, EMA 等指标

**审计与事件流**:
- **不可变事件流**: `order_events` 记录订单全生命周期（CREATED -> SUBMITTED -> FILLED -> CANCELED）
- **持仓快照**: `position_snapshots` 每 5 分钟或关键事件触发，用于复盘浮盈回撤与保证金变化

**自动归档**:
- 仅保留最近 90 天的热数据
- 每日凌晨运行归档任务，将旧数据移动至 `_history` 表

### 13.4 资金管理与风控

**动态"地板"保证金**:
- 核心逻辑: 单笔交易 **Initial Margin (初始保证金)** ≥ **50 USDT**
- 计算公式: `Base_Margin = Equity * 10%`, `Final_Margin = max(50, Base_Margin)`
- AI 增强: 若 AI Score > 85，系数可放大至 1.2
- 风控约束: 若 `Final_Margin * 杠杆 * 止损距离% > Total_Equity * 3%`，系统优先**降低杠杆**

**强平距离硬约束**:
- 模式: **强制逐仓 (Isolated Margin)**
- 杠杆范围: **10x - 20x** (动态调整)
- 硬性检查: `|Entry - Liq_Price| > 1.5 × |Entry - Stop_Loss|`

### 13.5 策略核心逻辑

**趋势与动量过滤**:
1. **ADX 趋势强度:** `ADX(14) > 25` 且 `+DI > -DI` (做多)
2. **挤压动量 (Squeeze Momentum)**:
   - **Squeeze Release:** 状态由 `On` (挤压) 变为 `Off` (释放)
   - **Momentum:** 柱状图由负转正 (做多)
3. **成交量确认:** `Current_Vol > 1.5 * MA(Vol, 5)`

**入场模型组合 (15m Timeframe)**:
- **Setup A (Trend Pullback)**:
  - 环境: ADX > 25 (强趋势)
  - 触发: 回踩 EMA21/55 + 拒绝K线 (Wick) + 吞没形态
- **Setup B (Squeeze Breakout)**:
  - 环境: Squeeze Release + Momentum 翻红为绿 + 量能爆发

### 13.6 AI 智能大脑

- **模型:** `SGDClassifier` (Scikit-learn, 支持增量学习)
- **数据源:** `market_data_cache` (特征) + `trade_logs` (标签)
- **特征工程:** `ADX_Value`, `Squeeze_Status` (0/1/2), `Vol_Ratio`, `BTC_Correlation`, `RSI_Slope`
- **训练机制:** 每次平仓 (Close Event) 触发一次 `partial_fit` 更新模型权重

### 13.7 管理与交互

**Admin Tool (CLI)**:
通过 Docker exec 运行管理工具，无需重启容器：
```bash
python -m scripts.trading_test_tool status          # 查看进程与持仓
python -m scripts.trading_test_tool set             # 热修改参数
python -m scripts.trading_test_tool emergency-exit  # 一键清仓
```
