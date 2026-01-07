#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/restore_db.sh backup.sql.gz
#
# Requires env:
#   DB_HOST DB_PORT DB_USER DB_PASS DB_NAME

IN="${1:-}"
if [[ -z "$IN" ]]; then
  echo "Usage: bash scripts/restore_db.sh <backup.sql.gz>" >&2
  exit 2
fi
if [[ ! -f "$IN" ]]; then
  echo "File not found: $IN" >&2
  exit 2
fi

: "${DB_HOST:=mariadb}"
: "${DB_PORT:=3306}"
: "${DB_USER:=root}"
: "${DB_PASS:=root}"
: "${DB_NAME:=alpha_sniper}"

echo "Restoring ${IN} -> ${DB_HOST}:${DB_PORT}/${DB_NAME}"
gunzip -c "$IN" | mysql -h"${DB_HOST}" -P"${DB_PORT}" -u"${DB_USER}" -p"${DB_PASS}" "${DB_NAME}"
echo "OK"
