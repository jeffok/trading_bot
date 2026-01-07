#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DB_HOST=... DB_PORT=... DB_USER=... DB_PASS=... DB_NAME=...
#   bash scripts/backup_db.sh [output.sql.gz]
#
# If output file is not provided, it will write to ./backups/alpha_sniper_<timestamp>.sql.gz

OUT="${1:-}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DIR="./backups"
mkdir -p "$DIR"
if [[ -z "$OUT" ]]; then
  OUT="$DIR/alpha_sniper_${TS}.sql.gz"
fi

: "${DB_HOST:=mariadb}"
: "${DB_PORT:=3306}"
: "${DB_USER:=root}"
: "${DB_PASS:=root}"
: "${DB_NAME:=alpha_sniper}"

echo "Backing up ${DB_HOST}:${DB_PORT}/${DB_NAME} -> ${OUT}"
mysqldump -h"${DB_HOST}" -P"${DB_PORT}" -u"${DB_USER}" -p"${DB_PASS}"   --single-transaction --routines --triggers --events   "${DB_NAME}" | gzip -9 > "${OUT}"
echo "OK: ${OUT}"
