"""幂等键与 client_order_id 生成（V8.3 对齐）。

文档口径（4.1.2）：
  client_order_id = asv8-{symbol}-{side}-{timeframe}-{bar_close_ts}-{nonce}

设计目标：
- 同一“交易机会/同一意图”的重试必须生成相同的 client_order_id（幂等）。
- 保持可读性，同时控制长度（常见限制 <= 64）。

说明：
- symbol：做 normalize（去掉 / - : 空格 等）
- side：BUY / SELL
- timeframe：例如 15m（由 interval_minutes 推导）
- bar_close_ts：毫秒时间戳（kline_open_time_ms + interval_ms）
- nonce：用于区分同一根 K 内不同动作；默认从 trace_id 派生稳定短 hash（保证重试一致）
"""

from __future__ import annotations

import hashlib
from typing import Optional


def normalize_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    for ch in ["/", "-", ":", " "]:
        s = s.replace(ch, "")
    return s


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[: max(4, int(n))]


def make_client_order_id_v83(
    *,
    symbol: str,
    side: str,
    interval_minutes: int,
    kline_open_time_ms: int,
    nonce: str,
    max_len: int = 64,
) -> str:
    """V8.3 口径的 client_order_id 生成器。"""
    sym = normalize_symbol(symbol)
    sd = (side or "").upper().strip()
    tf = f"{int(interval_minutes)}m"
    close_ts = int(kline_open_time_ms) + int(interval_minutes) * 60_000
    nn = (nonce or "0").strip()
    base = f"asv8-{sym}-{sd}-{tf}-{close_ts}-{nn}"
    if len(base) <= max_len:
        return base

    # 超长：缩短 symbol + hash 保持唯一
    sym_short = sym[:10]
    h = _short_hash(base, 10)
    short = f"asv8-{sym_short}-{sd}-{tf}-{close_ts}-{h}"
    return short[:max_len]


def make_client_order_id(
    action: str,
    symbol: str,
    *,
    interval_minutes: int,
    kline_open_time_ms: int,
    trace_id: Optional[str] = None,
    max_len: int = 64,
) -> str:
    """兼容旧调用点的包装器（映射到 V8.3 格式）。"""
    a = (action or "").lower().strip()
    side = "BUY" if a in ("buy", "open", "long") else "SELL"
    base_nonce = _short_hash(trace_id or f"{symbol}-{kline_open_time_ms}", 8)
    nonce = f"{a[:2]}{base_nonce}"
    return make_client_order_id_v83(
        symbol=symbol,
        side=side,
        interval_minutes=int(interval_minutes),
        kline_open_time_ms=int(kline_open_time_ms),
        nonce=nonce,
        max_len=max_len,
    )
