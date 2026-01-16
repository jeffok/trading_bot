
"""Simple migrations runner."""

from __future__ import annotations
import re
from pathlib import Path
from typing import List
from .postgres import PostgreSQL

MIGRATION_RE = re.compile(r"^(\d{4})_.*\.sql$")

def migrate(db: PostgreSQL, migrations_dir: Path) -> List[str]:
    migrations_dir = migrations_dir.resolve()
    with db.tx() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version VARCHAR(32) PRIMARY KEY,
              applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
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

        raw_sql = p.read_text(encoding="utf-8")

        # IMPORTANT:
        # Migration files may contain comments that include semicolons.
        # A naive `split(';')` breaks those comments into executable fragments
        # and can cause SQL syntax errors during startup.
        #
        # We implement a lightweight sanitizer:
        # - remove /* ... */ blocks
        # - remove full-line `-- ...` comments
        # - remove inline `-- ...` comments (best-effort)
        # Then split by semicolon.

        # Remove C-style block comments.
        sql = re.sub(r"/\*.*?\*/", "", raw_sql, flags=re.DOTALL)

        cleaned_lines = []
        for line in sql.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("--"):
                continue
            # Remove inline `-- comment` if present.
            # MySQL requires a space after `--` for comment, but we treat any `--` as comment starter.
            if "--" in line:
                line = line.split("--", 1)[0]
            if line.strip():
                cleaned_lines.append(line)

        sql = "\n".join(cleaned_lines)
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        with db.tx() as cur:
            for st in statements:
                cur.execute(st)
            cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))
        ran.append(p.name)

    return ran
