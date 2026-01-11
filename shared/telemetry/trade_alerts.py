from __future__ import annotations

from typing import Any, Dict, Optional

def _val(v: Any) -> Any:
    if v is None:
        return None
    # Enum support
    if hasattr(v, "value"):
        try:
            return getattr(v, "value")
        except Exception:
            return str(v)
    return v

def _round(v: Any, nd: int) -> Any:
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except Exception:
        return v

def _trim_text(v: Any, max_len: int = 240) -> Any:
    if v is None:
        return None
    try:
        s = str(v)
    except Exception:
        return v
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s

def build_trade_summary(
    *,
    event: str,
    trace_id: str,
    exchange: str,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    qty: Any = None,
    price: Any = None,
    leverage: Any = None,
    ai_score: Any = None,
    stop_price: Any = None,
    stop_dist_pct: Any = None,
    reason_code: Any = None,
    reason: Any = None,
    client_order_id: Optional[str] = None,
    exchange_order_id: Optional[str] = None,
    stop_client_order_id: Optional[str] = None,
    stop_exchange_order_id: Optional[str] = None,
    status: Optional[str] = None,
    error: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """统一 trade 类 Telegram summary_kv 字段口径（V8.3：字段尽量齐全一致）。"""
    kv: Dict[str, Any] = {
        "event": event,
        "trace_id": trace_id,
        "exchange": exchange,
        "symbol": symbol,
        "side": side,
        "qty": _round(qty, 6),
        "price": _round(price, 6),
        "leverage": _val(leverage),
        "ai_score": _round(ai_score, 2),
        "stop_price": _round(stop_price, 6),
        "stop_dist_pct": _round(stop_dist_pct, 6),
        "reason_code": _val(reason_code),
        "reason": _trim_text(reason),
        "client_order_id": client_order_id,
        "exchange_order_id": exchange_order_id,
        "stop_client_order_id": stop_client_order_id,
        "stop_exchange_order_id": stop_exchange_order_id,
        "status": status,
        "error": _trim_text(error),
    }
    if extra:
        for k, v in extra.items():
            kv[k] = v
    # drop None
    return {k: v for k, v in kv.items() if v is not None}

def send_trade_alert(
    telegram,
    *,
    title: str,
    summary_kv: Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """统一 trade 告警发送入口，避免各处字段缺失/不一致。"""
    telegram.send_alert_zh(title=title, summary_kv=summary_kv, payload=payload or {})
