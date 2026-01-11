#!/usr/bin/env bash
set -euo pipefail

# Drill (B2): Update protective-stop runtime configs in system_config and verify /admin/status reflects them.
# Usage:
#   ADMIN_TOKEN=xxx ./scripts/drills/update_stop_config_runtime.sh 3 0.5 2 60
#   # args: STOP_ARM_MAX_RETRIES STOP_ARM_BACKOFF_BASE_SECONDS STOP_REARM_MAX_ATTEMPTS STOP_REARM_COOLDOWN_SECONDS

API_BASE="${API_BASE:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-change_me}"

ARM_RETRIES="${1:-3}"
ARM_BACKOFF="${2:-0.5}"
REARM_MAX="${3:-2}"
REARM_COOLDOWN="${4:-60}"

echo "[1/4] Update STOP_ARM_MAX_RETRIES=${ARM_RETRIES}"
curl -sS -X POST "${API_BASE}/admin/update_config"   -H "Content-Type: application/json"   -H "X-Admin-Token: ${ADMIN_TOKEN}"   -d "{\"key\":\"STOP_ARM_MAX_RETRIES\",\"value\":\"${ARM_RETRIES}\",\"actor\":\"drill\",\"reason_code\":\"DRILL\",\"reason\":\"update stop retries\"}" | jq .

echo "[2/4] Update STOP_ARM_BACKOFF_BASE_SECONDS=${ARM_BACKOFF}"
curl -sS -X POST "${API_BASE}/admin/update_config"   -H "Content-Type: application/json"   -H "X-Admin-Token: ${ADMIN_TOKEN}"   -d "{\"key\":\"STOP_ARM_BACKOFF_BASE_SECONDS\",\"value\":\"${ARM_BACKOFF}\",\"actor\":\"drill\",\"reason_code\":\"DRILL\",\"reason\":\"update stop backoff\"}" | jq .

echo "[3/4] Update STOP_REARM_MAX_ATTEMPTS=${REARM_MAX}, STOP_REARM_COOLDOWN_SECONDS=${REARM_COOLDOWN}"
curl -sS -X POST "${API_BASE}/admin/update_config"   -H "Content-Type: application/json"   -H "X-Admin-Token: ${ADMIN_TOKEN}"   -d "{\"key\":\"STOP_REARM_MAX_ATTEMPTS\",\"value\":\"${REARM_MAX}\",\"actor\":\"drill\",\"reason_code\":\"DRILL\",\"reason\":\"update stop rearm max\"}" | jq .

curl -sS -X POST "${API_BASE}/admin/update_config"   -H "Content-Type: application/json"   -H "X-Admin-Token: ${ADMIN_TOKEN}"   -d "{\"key\":\"STOP_REARM_COOLDOWN_SECONDS\",\"value\":\"${REARM_COOLDOWN}\",\"actor\":\"drill\",\"reason_code\":\"DRILL\",\"reason\":\"update stop rearm cooldown\"}" | jq .

echo "[4/4] Check /admin/status"
curl -sS "${API_BASE}/admin/status"   -H "X-Admin-Token: ${ADMIN_TOKEN}" | jq '.config | {USE_PROTECTIVE_STOP_ORDER, STOP_ORDER_POLL_SECONDS, STOP_ARM_MAX_RETRIES, STOP_ARM_BACKOFF_BASE_SECONDS, STOP_REARM_MAX_ATTEMPTS, STOP_REARM_COOLDOWN_SECONDS}'
