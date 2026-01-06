
from __future__ import annotations
import time
from typing import Dict, List, Optional
from .base import ExchangeClient
from .types import Kline, OrderResult

class PaperExchange(ExchangeClient):
    name = "paper"

    def __init__(self, starting_usdt: float = 1000.0, fee_pct: float = 0.0004):
        self.usdt = starting_usdt
        self.base = 0.0
        self.last_price: Dict[str, float] = {}
        self.fee_pct = fee_pct

    def update_last_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = float(price)

    def fetch_klines(self, *, symbol: str, interval_minutes: int, start_ms: Optional[int], limit: int = 1000) -> List[Kline]:
        return []

    def place_market_order(self, *, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        px = float(self.last_price.get(symbol, 0.0))
        fee = qty * px * self.fee_pct
        if side.upper() == "BUY":
            cost = qty * px + fee
            if px > 0 and cost > self.usdt:
                qty = max(0.0, (self.usdt / px) * (1 - self.fee_pct))
                cost = qty * px + qty * px * self.fee_pct
            self.usdt -= cost
            self.base += qty
        else:
            qty = min(qty, self.base)
            proceeds = qty * px - fee
            self.base -= qty
            self.usdt += max(0.0, proceeds)

        oid = f"paper_{int(time.time()*1000)}"
        return OrderResult(exchange_order_id=oid, status="FILLED", filled_qty=qty, avg_price=px, raw={"usdt": self.usdt, "base": self.base})

    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        return OrderResult(exchange_order_id=exchange_order_id or "", status="FILLED", filled_qty=0.0)
