#!/usr/bin/env bash
set -euo pipefail

# Arm protective stop order demo:
# 1) Set system_config USE_PROTECTIVE_STOP_ORDER=true
# 2) Start services
# 3) Trigger an open (or use RUN_ONCE=true)
# 4) Verify stop order ids stored in trade_logs and position meta

ADMIN_TOKEN="${ADMIN_TOKEN:-change_me}"
API_BASE="${API_BASE:-http://localhost:8000}"

echo "[1/3] Enable protective stop (system_config)"
curl -sS -X POST "$API_BASE/admin/config/system_config" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"key":"USE_PROTECTIVE_STOP_ORDER","value":"true"}' | jq .

echo "[2/3] Set stop poll seconds"
curl -sS -X POST "$API_BASE/admin/config/system_config" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"key":"STOP_ORDER_POLL_SECONDS","value":"10"}' | jq .

echo "[3/3] Check admin status"
curl -sS "$API_BASE/admin/status" -H "X-Admin-Token: $ADMIN_TOKEN" | jq '.config | {USE_PROTECTIVE_STOP_ORDER, STOP_ORDER_POLL_SECONDS}'
