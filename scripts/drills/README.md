# Drills (Milestone C)

These are **non-destructive** operational drills to validate:
- service heartbeats are updated (`service_status`)
- `/health` reflects dependency availability
- workers continue after restart (idempotency + best-effort reconcile)

## Quick start

```bash
docker compose up -d --build
bash scripts/drills/restart_strategy_engine.sh
bash scripts/drills/restart_data_syncer.sh
```

## Metrics

- API: `http://localhost:8080/metrics`
- data-syncer: `http://localhost:9101/`
- strategy-engine: `http://localhost:9102/`


## E2E 演练（里程碑 E）

一键跑通：建库/迁移 →（paper 模式自动造数）→ 跑一次 data-syncer → 跑一次 strategy-engine → 打印 trade_logs / order_events

```bash
bash scripts/drills/e2e_trade_cycle.sh
```

说明：
- 当 `EXCHANGE=paper` 时，会用 `scripts/drills/seed_synthetic_data.py` 生成 K 线与 features，方便离线验证 AI/策略/止损/审计链路。
- 当 `EXCHANGE=binance/bybit` 时，会直接跑一次 data-syncer 拉真实 K 线（需要网络与 API 配置）。


## DB 备份/恢复（里程碑 F）
- `bash scripts/backup_db.sh`：导出并压缩到 `./backups/`
- `bash scripts/restore_db.sh <file.sql.gz>`：从备份恢复

> 需要本机/容器内可用 `mysqldump` 与 `mysql` 客户端。生产环境建议在单独维护容器或运维机上运行。


## A2: SYMBOLS 热更新演练

- 修改 system_config 的 `SYMBOLS` 并验证 `/admin/status` 的 `EFFECTIVE_SYMBOLS` 更新：

```bash
ADMIN_TOKEN=change_me ./scripts/drills/update_symbols_runtime.sh "BTCUSDT,ETHUSDT,SOLUSDT"
```

> data-syncer / strategy-engine 会在 `RUNTIME_CONFIG_REFRESH_SECONDS`（默认 30s）内拾取新 symbols。


## B2: 保护止损运行时参数更新演练

```bash
ADMIN_TOKEN=change_me ./scripts/drills/update_stop_config_runtime.sh 3 0.5 2 60
```

> strategy-engine 会在 `RUNTIME_CONFIG_REFRESH_SECONDS`（默认 30s）内拾取新参数。
