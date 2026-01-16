#!/usr/bin/env bash
set -euo pipefail

# E2E drill:
# - start infra (postgres/redis)
# - migrate schema
# - (paper mode) seed synthetic klines+features
# - run data-syncer once (optional)
# - run strategy-engine once (RUN_ONCE=true)
# - print last trade_logs / order_events

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"

if [ ! -f docker-compose.yml ]; then
  echo "[drill] docker-compose.yml not found -> copy from docker-compose.yml.example"
  cp docker-compose.yml.example docker-compose.yml
fi

if [ ! -f .env ]; then
  echo "[drill] .env not found -> copy from .env.example"
  cp .env.example .env
fi

set -a
source .env
set +a

echo "[drill] up infra..."
docker compose up -d postgres redis

echo "[drill] migrate (via data-syncer container)..."
docker compose run --rm -e RUN_ONCE=true data-syncer >/dev/null

if [ "${EXCHANGE:-paper}" = "paper" ]; then
  echo "[drill] EXCHANGE=paper -> seed synthetic market data"
  docker compose run --rm data-syncer python scripts/drills/seed_synthetic_data.py
else
  echo "[drill] EXCHANGE=${EXCHANGE} -> run data-syncer once to pull real klines"
  docker compose run --rm -e RUN_ONCE=true data-syncer
fi

echo "[drill] run strategy-engine once..."
docker compose run --rm -e RUN_ONCE=true strategy-engine

# Print results (使用trading_test_tool查询)
echo "[drill] show last trade_logs..."
docker compose exec execution python -m scripts.trading_test_tool query --sql "SELECT id,created_at,symbol,status,qty,entry_price,exit_price,pnl,robot_score,ai_prob,open_reason_code,close_reason_code FROM trade_logs ORDER BY id DESC LIMIT 5" || true

echo "[drill] show last order_events..."
docker compose exec execution python -m scripts.trading_test_tool query --sql "SELECT id,created_at,symbol,client_order_id,event_type,side,qty,price,status,reason_code FROM order_events ORDER BY id DESC LIMIT 10" || true

echo "[drill] done."
