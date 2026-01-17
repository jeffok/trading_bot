
# Alpha-Sniper-V8 (B-lite) - MVP Reference Implementation

这是一个按照你提供的 V8.3 规格文档实现的 **可运行 MVP** 代码库骨架，目标是：

- 三服务架构：`data-syncer` / `strategy-engine` / `api-service`
- 交易所：支持 **Binance** 与 **Bybit**（运行时二选一，不会同时使用多个交易所）
- 核心不变量：幂等下单（`client_order_id` / `orderLinkId`）、事件流 `order_events`、分布式锁、可观测性（/health /metrics）、可恢复（对账补写）
- 数据：PostgreSQL（强一致事件与历史，JSONB格式存储）、Redis（锁/速率限制状态）

> ⚠️ 交易有风险。默认建议先使用 `EXCHANGE=paper`（纸交易）验证端到端流程，再切换真实交易所并设置较小仓位。

---

## 1. 快速启动（Docker Compose）

1) 复制环境变量模板并修改：

```bash
cp .env.example .env
```

2) 启动：

```bash
docker compose up --build
```

3) 检查健康：

- API：`http://localhost:9001/health`
- Metrics：`http://localhost:9001/metrics`

---

## 2. 运行时选择交易所（二选一）

在 `.env` 中设置：

- `EXCHANGE=binance` 或 `EXCHANGE=bybit`（或 `paper`）
- 若 `binance`：提供 `BINANCE_API_KEY/BINANCE_API_SECRET`
- 若 `bybit`：提供 `BYBIT_API_KEY/BYBIT_API_SECRET`

本项目在正常运行模式下 **只会选择一个交易所客户端**，不会同时调用多个交易所。

---

## 3. 项目结构（简述）

- `shared/`：公共库（配置、日志、DB、Redis 锁、交易所网关、领域模型、指标与 Telegram）
- `services/`：三个服务
- `scripts/trading_test_tool/`：命令行管理工具（仅在Docker中使用，可替代或补充 /admin）
- `migrations/`：PostgreSQL 初始化与升级脚本（自动执行）

---

## 4. 重要约定（强制）

- **时区**：调度按 `Asia/Hong_Kong` 计算 tick；DB 存储时间统一用 UTC。
- **幂等**：所有下单都必须携带 `client_order_id`（Binance: `newClientOrderId` / Bybit: `orderLinkId`）。
- **事件流**：订单生命周期必须写入 `order_events`，用于审计/恢复/对账。
- **止损一致性**：触发止损必须写事件并 Telegram 通知（带 `trace_id` 与 `reason_code`）。

---

## 5. 常用管理接口

API 服务：`http://localhost:9001`

- `POST /admin/halt`  （暂停策略下单）
- `POST /admin/resume`（恢复策略下单）
- `POST /admin/emergency_exit`（紧急平仓/卖出）
- `POST /admin/update_config`（写 system_config，带审计）

认证：`Authorization: Bearer <ADMIN_TOKEN>`

---

## 6. Trading Test Tool（仅在Docker中使用）

```bash
# 准备检查（检查配置、服务状态等）
docker compose exec api-service tbot prepare

# 查看系统状态
docker compose exec api-service tbot status

# 诊断为什么没有下单
docker compose exec api-service tbot diagnose

# 暂停/恢复交易
docker compose exec api-service tbot halt --by admin --reason-code ADMIN_HALT --reason "maintenance"
docker compose exec api-service tbot resume --by admin --reason-code ADMIN_RESUME --reason "ok"
```

> ⚠️ 重要：此工具只能在Docker容器中使用。CLI 会直接读写 PostgreSQL（等价于调用 /admin 的写入路径），同样会写审计与可选 Telegram。

---

## 7. 开发说明

- 本仓库使用 `requirements.txt` 以降低环境复杂度。
- 每个模块都有较详细注释，便于你后续扩展策略/AI/回测。
