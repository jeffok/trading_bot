#!/usr/bin/env bash
set -euo pipefail

# 创建数据库脚本
# 用法: ./scripts/create_db.sh

echo "[create_db] Creating database if not exists..."

python - <<'PY'
import os
import sys
from urllib.parse import urlparse, urlunparse

postgres_url = os.getenv("POSTGRES_URL", "").strip()
if not postgres_url:
    print("[create_db] ERROR: POSTGRES_URL not set")
    sys.exit(1)

parsed = urlparse(postgres_url)
db_name = parsed.path.lstrip('/')

if not db_name:
    print("[create_db] ERROR: No database name in POSTGRES_URL")
    sys.exit(1)

# 连接到默认数据库 postgres 来创建目标数据库
admin_url = urlunparse((
    parsed.scheme,
    f"{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port or 5432}",
    "postgres",  # 连接到默认数据库
    parsed.params,
    parsed.query,
    parsed.fragment
))

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    conn = psycopg2.connect(admin_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # 检查数据库是否存在
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone()
    
    if exists:
        print(f"[create_db] Database '{db_name}' already exists")
    else:
        # 创建数据库
        cur.execute(f'CREATE DATABASE "{db_name}"')
        print(f"[create_db] Database '{db_name}' created successfully")
    
    cur.close()
    conn.close()
    print("[create_db] Done")
    sys.exit(0)
except ImportError:
    print("[create_db] ERROR: psycopg2 not installed")
    sys.exit(1)
except Exception as e:
    print(f"[create_db] ERROR: {e}")
    sys.exit(1)
PY
