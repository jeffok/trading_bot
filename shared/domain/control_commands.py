from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from shared.db import PostgreSQL


def write_control_command(
    db: PostgreSQL,
    *,
    command: str,
    payload: Dict[str, Any],
    status: str = "NEW",
    trace_id: Optional[str] = None,
    actor: Optional[str] = None,
    reason_code: Optional[str] = None,
    reason: Optional[str] = None,
) -> int:
    """Append a control command (auditable). Returns inserted id (best-effort)."""
    try:
        # PostgreSQL: 使用RETURNING获取插入的ID
        row = db.fetch_one(
            """
            INSERT INTO control_commands(command, payload_json, status, trace_id, actor, reason_code, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                str(command),
                json.dumps(payload, ensure_ascii=False),
                str(status),
                (str(trace_id) if trace_id else None),
                (str(actor) if actor else None),
                (str(reason_code) if reason_code else None),
                (str(reason) if reason else None),
            ),
        )
        return int(row["id"]) if row and row.get("id") is not None else 0
    except Exception:
        return 0


def fetch_new_control_commands(db: PostgreSQL, *, limit: int = 50) -> List[Dict[str, Any]]:
    rows = db.fetch_all(
        """
        SELECT id, created_at, command, payload_json, trace_id, actor, reason_code, reason
        FROM control_commands
        WHERE status='NEW'
        ORDER BY id ASC
        LIMIT %s
        """,
        (int(limit),),
    )
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        payload = {}
        try:
            payload = json.loads(r.get("payload_json") or "{}")
        except Exception:
            payload = {}
        out.append(
            {
                "id": int(r["id"]),
                "created_at": r.get("created_at"),
                "command": str(r.get("command") or ""),
                "payload": payload,
                "trace_id": str(r.get("trace_id") or "") if r.get("trace_id") is not None else None,
                "actor": str(r.get("actor") or "") if r.get("actor") is not None else None,
                "reason_code": str(r.get("reason_code") or "") if r.get("reason_code") is not None else None,
                "reason": str(r.get("reason") or "") if r.get("reason") is not None else None,
            }
        )
    return out


def mark_control_command_processed(
    db: PostgreSQL,
    *,
    command_id: int,
    status: str = "PROCESSED",
) -> None:
    db.execute(
        """
        UPDATE control_commands
        SET status=%s, processed_at=CURRENT_TIMESTAMP
        WHERE id=%s
        """,
        (str(status), int(command_id)),
    )
