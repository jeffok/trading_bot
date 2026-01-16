from __future__ import annotations

import datetime
import json
from typing import Any, Dict

from ..db import PostgreSQL


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return obj


def upsert_service_status(db: PostgreSQL, *, service_name: str, instance_id: str, status: Dict[str, Any]) -> None:
    """Upsert service heartbeat into service_status (append-only elsewhere; this is a mutable heartbeat table)."""
    payload = json.dumps(status, ensure_ascii=False, default=_jsonable)
    db.execute(
        """
        INSERT INTO service_status(service_name, instance_id, last_heartbeat, status_json)
        VALUES (%s,%s,CURRENT_TIMESTAMP,%s)
        ON CONFLICT (service_name, instance_id) DO UPDATE SET last_heartbeat=CURRENT_TIMESTAMP, status_json=EXCLUDED.status_json
        """,
        (service_name, instance_id, payload),
    )
