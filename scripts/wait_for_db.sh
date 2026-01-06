#!/usr/bin/env bash
set -euo pipefail

host="${DB_HOST:-mariadb}"
port="${DB_PORT:-3306}"

echo "[wait_for_db] waiting for ${host}:${port} ..."
for i in $(seq 1 60); do
  if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
    echo "[wait_for_db] DB port is open."
    exit 0
  fi
  sleep 1
done

echo "[wait_for_db] DB not reachable after timeout."
exit 1
