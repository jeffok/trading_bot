from __future__ import annotations

import datetime
import json
from typing import Any, Dict


def log_action(logger, action: str, **fields: Any) -> None:
    """轻量结构化日志（不引入额外依赖）。

    需求口径：action / reason_code / reason / trace_id / client_order_id 等字段必须可被检索。
    这里直接输出一行 JSON，方便 grep / Loki / ELK 解析。
    """
    record: Dict[str, Any] = {
        "ts_utc": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
        "action": action,
    }
    record.update(fields or {})
    try:
        logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        # fallback: avoid breaking trading loop
        logger.info(f"[action={action}] {record}")
