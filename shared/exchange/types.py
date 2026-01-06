
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass(frozen=True)
class OrderResult:
    exchange_order_id: str
    status: str
    filled_qty: float
    avg_price: Optional[float] = None
    raw: Optional[dict] = None
