from __future__ import annotations

from typing import Any, Dict, Optional

def _val(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "value"):
        try:
            return getattr(v, "value")
        except Exception:
            return str(v)
    return v

def build_system_summary(
    *,
    event: str,
    trace_id: str,
    level: str = "INFO",
    actor: Optional[str] = None,
    exchange: Optional[str] = None,
    reason_code: Any = None,
    reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """统一系统类 Telegram summary_kv 字段口径（用于 HALT/RESUME/EMERGENCY/CONFIG 等）。"""
    kv: Dict[str, Any] = {
        "level": level,
        "event": event,
        "trace_id": trace_id,
        "actor": actor,
        "exchange": exchange,
        "reason_code": _val(reason_code),
        "reason": reason,
    }
    if extra:
        for k, v in extra.items():
            kv[k] = v
    return {k: v for k, v in kv.items() if v is not None}

def send_system_alert(
    telegram,
    *,
    title: str,
    summary_kv: Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """统一系统类告警发送入口。"""
    telegram.send_alert_zh(title=title, summary_kv=summary_kv, payload=payload or {})
