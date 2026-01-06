
"""Simple migrations runner."""

from __future__ import annotations
import re
from pathlib import Path
from typing import List
from .maria import MariaDB

MIGRATION_RE = re.compile(r"^(\d{4})_.*\.sql$")

def migrate(db: MariaDB, migrations_dir: Path) -> List[str]:
    migrations_dir = migrations_dir.resolve()
    with db.tx() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version VARCHAR(32) PRIMARY KEY,
              applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )

    applied = {r["version"] for r in db.fetch_all("SELECT version FROM schema_migrations")}
    ran: List[str] = []

    for p in sorted(migrations_dir.glob("*.sql")):
        m = MIGRATION_RE.match(p.name)
        if not m:
            continue
        version = m.group(1)
        if version in applied:
            continue

        sql = p.read_text(encoding="utf-8")
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        with db.tx() as cur:
            for st in statements:
                cur.execute(st)
            cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))
        ran.append(p.name)

    return ran
