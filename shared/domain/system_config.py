from __future__ import annotations

from shared.db import PostgreSQL


def get_system_config(db: PostgreSQL, key: str, default: str | None = None) -> str:
    row = db.fetch_one('SELECT "value" FROM system_config WHERE "key"=%s', (key,))
    if row and row.get("value") is not None:
        return str(row["value"])
    return "" if default is None else str(default)


def write_system_config(
    db: PostgreSQL,
    *,
    actor: str,
    key: str,
    value: str,
    trace_id: str,
    reason_code: str,
    reason: str,
    action: str = "SET",
) -> None:
    old = db.fetch_one('SELECT "value" FROM system_config WHERE "key"=%s', (key,))
    old_val = old["value"] if old else None

    db.execute(
        'INSERT INTO "system_config"("key","value") VALUES (%s,%s) ON CONFLICT ("key") DO UPDATE SET "value"=EXCLUDED."value"',
        (key, value),
    )
    db.execute(
        """
        INSERT INTO config_audit(actor, action, cfg_key, old_value, new_value, trace_id, reason_code, reason)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (actor, action, key, old_val, value, trace_id, reason_code, reason),
    )
