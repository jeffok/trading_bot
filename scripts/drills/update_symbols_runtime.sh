#!/usr/bin/env bash
set -euo pipefail

# Drill: Update SYMBOLS in system_config at runtime and verify /admin/status reflects the change.
# Requirements:
# - api-service is running and reachable (default http://localhost:8000)
# - ADMIN_TOKEN is set (or passed via env)
#
# Usage:
#   ADMIN_TOKEN=xxx ./scripts/drills/update_symbols_runtime.sh "BTCUSDT,ETHUSDT,SOLUSDT"
#
# Tips:
# - The change will be picked up by data-syncer / strategy-engine within RUNTIME_CONFIG_REFRESH_SECONDS (default 30s).

API_BASE="${API_BASE:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-change_me}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 "BTCUSDT,ETHUSDT,SOLUSDT""
  exit 1
fi

NEW_SYMBOLS="$1"

echo "[1/3] Update system_config SYMBOLS=${NEW_SYMBOLS}"
curl -sS -X POST "${API_BASE}/admin/update_config" \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}" \
  -d "{"key":"SYMBOLS","value":"${NEW_SYMBOLS}","actor":"drill","reason_code":"DRILL","reason":"update symbols runtime"}" | jq .

echo "[2/3] Check /admin/status"
curl -sS "${API_BASE}/admin/status" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}" | jq '.config | {EXCHANGE, EFFECTIVE_SYMBOLS, SYMBOLS_FROM_DB, HALT_TRADING, EMERGENCY_EXIT}'

echo "[3/3] Next: observe services picking the change within refresh window"
echo "    docker compose logs -f data-syncer strategy-engine | egrep "runtime_config|symbols_updated|start service""
