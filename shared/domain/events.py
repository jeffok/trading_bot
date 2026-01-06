
"""Append-only order event writer (idempotent)."""

from __future__ import annotations
import json
from typing import Any, Dict, Optional
from ..db.maria import MariaDB
from .enums import OrderEventType, ReasonCode

def append_order_event(
    db: MariaDB,
    *,
    trace_id: str,
    service: str,
    exchange: str,
    symbol: str,
    client_order_id: str,
    exchange_order_id: Optional[str],
    event_type: OrderEventType,
    side: str,
    qty: float,
    price: Optional[float],
    status: str,
    reason_code: ReasonCode,
    reason: str,
    payload: Dict[str, Any],
) -> None:
    sql = """
    INSERT INTO order_events(
      trace_id, service, exchange, symbol, client_order_id, exchange_order_id,
      event_type, side, qty, price, status, reason_code, reason, payload_json
    ) VALUES (
      %s,%s,%s,%s,%s,%s,
      %s,%s,%s,%s,%s,%s,%s,%s
    )
    """
    params = (
        trace_id, service, exchange, symbol, client_order_id, exchange_order_id,
        event_type.value, side, float(qty), float(price) if price is not None else None,
        status, reason_code.value, reason, json.dumps(payload, ensure_ascii=False),
    )
    try:
        db.execute(sql, params)
    except Exception as e:
        msg = str(e).lower()
        if "duplicate" in msg and "uq_client_order" in msg:
            return
        raise
