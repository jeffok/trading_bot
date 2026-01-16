from __future__ import annotations

import json
from typing import Any, Dict, Optional

from shared.db import PostgreSQL


def load_current_model_blob(db: PostgreSQL, *, model_name: str) -> Optional[Dict[str, Any]]:
    """Load current model dict from ai_models (is_current=1)."""
    row = db.fetch_one(
        """
        SELECT id, model_name, version, metrics_json, blob
        FROM ai_models
        WHERE model_name=%s AND is_current=TRUE
        ORDER BY id DESC
        LIMIT 1
        """,
        (str(model_name),),
    )
    if not row:
        return None
    blob = row.get('blob')
    if not blob:
        return None
    try:
        if isinstance(blob, (bytes, bytearray)):
            s = blob.decode('utf-8', errors='ignore')
        else:
            s = str(blob)
        s = s.strip()
        if not s:
            return None
        d = json.loads(s)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def save_current_model_blob(
    db: PostgreSQL,
    *,
    model_name: str,
    version: str,
    model_dict: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a model dict into ai_models and mark it as current (best-effort)."""
    try:
        payload = json.dumps(model_dict or {}, ensure_ascii=False).encode('utf-8')
        metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
        with db.tx() as cur:
            cur.execute(
                'UPDATE ai_models SET is_current=FALSE WHERE model_name=%s AND is_current=TRUE',
                (str(model_name),),
            )
            cur.execute(
                'INSERT INTO ai_models(model_name, version, is_current, metrics_json, blob) VALUES (%s,%s,TRUE,%s,%s)',
                (str(model_name), str(version), metrics_json, payload),
            )
    except Exception:
        return
