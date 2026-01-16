"""Append-only order event writer (idempotent)."""

from __future__ import annotations

import datetime
import json
from decimal import Decimal
from typing import Any, Dict, Optional, Union

from ..db import PostgreSQL
from .enums import OrderEventType, ReasonCode


def _json_default(o: Any) -> Any:
    """json.dumps fallback for Decimal / datetime."""
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        try:
            return float(o)
        except Exception:
            return str(o)
    return str(o)


def _utc_to_hk_naive(utc_dt: datetime.datetime) -> datetime.datetime:
    """Convert UTC datetime (naive or aware) to HK naive datetime (UTC+8)."""
    if utc_dt.tzinfo is not None:
        utc_dt = utc_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return utc_dt + datetime.timedelta(hours=8)


SENSITIVE_KEY_HINTS = (
    "api_key", "apikey", "secret", "signature", "sig",
    "token", "bearer", "authorization", "auth",
    "password", "passphrase", "private_key", "key",
)


def sanitize_payload(obj: Any, *, max_str_len: int = 2000, max_depth: int = 6, _depth: int = 0) -> Any:
    """Sanitize payload for storage (V8.3)."""
    if _depth >= max_depth:
        return "<max_depth_reached>"
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                ks = str(k).lower()
                if any(h in ks for h in SENSITIVE_KEY_HINTS):
                    out[k] = "***"
                else:
                    out[k] = sanitize_payload(v, max_str_len=max_str_len, max_depth=max_depth, _depth=_depth + 1)
            return out
        if isinstance(obj, list):
            return [sanitize_payload(v, max_str_len=max_str_len, max_depth=max_depth, _depth=_depth + 1) for v in obj[:200]]
        if isinstance(obj, (datetime.datetime, datetime.date, Decimal)):
            return _json_default(obj)
        if isinstance(obj, str):
            return obj if len(obj) <= max_str_len else (obj[:max_str_len] + "...<truncated>")
        return obj
    except Exception:
        return "<sanitize_failed>"


def get_first_event_created_at(
    db: PostgreSQL,
    *,
    exchange: str,
    symbol: str,
    client_order_id: str,
    event_type: OrderEventType = OrderEventType.CREATED,
) -> Optional[datetime.datetime]:
    row = db.fetch_one(
        """
        SELECT created_at
        FROM order_events
        WHERE exchange=%s AND symbol=%s AND client_order_id=%s AND event_type=%s
        ORDER BY id ASC
        LIMIT 1
        """,
        (exchange, symbol, client_order_id, event_type.value),
    )
    return row.get("created_at") if row else None


def append_order_event(
    db: PostgreSQL,
    *,
    trace_id: str,
    service: str,
    exchange: str,
    symbol: str,
    client_order_id: Optional[str],
    exchange_order_id: Optional[str],
    event_type: Union[OrderEventType, str],
    action: Optional[str] = None,
    actor: Optional[str] = None,
    side: str,
    qty: float,
    price: Optional[float],
    status: str,
    reason_code: Union[ReasonCode, str],
    reason: str,
    payload: Dict[str, Any],
) -> bool:
    """Insert into order_events (append-only + idempotent)."""
    et = event_type.value if isinstance(event_type, OrderEventType) else str(event_type or OrderEventType.ERROR.value)
    rc = reason_code.value if isinstance(reason_code, ReasonCode) else str(reason_code or ReasonCode.ERROR.value)

    act = (action or et).strip()[:64]
    actr = (actor or service).strip()[:64]
    utc_now = datetime.datetime.utcnow()
    hk_now = _utc_to_hk_naive(utc_now)

    coid = (client_order_id or "").strip()
    if not coid:
        coid = f"SYS-{trace_id}"[:64]

    payload_obj = sanitize_payload(payload or {})
    payload_json = json.dumps(payload_obj, ensure_ascii=False, default=_json_default)
    sql = """
    INSERT INTO order_events(
        trace_id, service, exchange, symbol, client_order_id, exchange_order_id,
        event_type, action, actor, side, qty, price, status, reason_code, reason, event_ts_hk, raw_payload_json, payload_json
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (
        trace_id,
        service,
        exchange,
        symbol,
        coid,
        exchange_order_id,
        et,
        act,
        actr,
        side,
        float(qty) if qty is not None else 0.0,
        float(price) if price is not None else None,
        status,
        rc,
        str(reason or "")[:2000],
        hk_now,
        payload_json,
        payload_json,
    )
    try:
        db.execute(sql, params)
        return True
    except Exception as e:
        msg = str(e).lower()
        if "duplicate" in msg and ("uq_client_order_event" in msg or "uq_client_order" in msg):
            return False
        raise


def append_error_event(
    db: PostgreSQL,
    *,
    trace_id: str,
    service: str,
    exchange: str,
    symbol: str,
    reason: str,
    payload: Dict[str, Any],
    client_order_id: Optional[str] = None,
    reason_code: Union[ReasonCode, str] = ReasonCode.ERROR,
) -> None:
    """Best-effort system ERROR event."""
    try:
        append_order_event(
            db,
            trace_id=trace_id,
            service=service,
            exchange=exchange,
            symbol=symbol,
            client_order_id=client_order_id,
            exchange_order_id=None,
            event_type=OrderEventType.ERROR,
            side="SYSTEM",
            qty=0.0,
            price=None,
            status="ERROR",
            reason_code=reason_code,
            reason=reason,
            payload=payload or {},
        )
    except Exception:
        return
