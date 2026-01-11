from __future__ import annotations

import time
from typing import Dict, List, Optional

from .base import ExchangeClient
from .types import Kline, OrderResult


class PaperExchange(ExchangeClient):
    """A very small in-memory exchange for drills/tests.

    - Orders are filled immediately at last_price[symbol] (if available), otherwise 0.
    - Stop orders are accepted but never auto-triggered (strategy-engine still enforces stop logic in paper mode).
    """

    name = "paper"

    def __init__(self, starting_usdt: float = 1000.0, fee_pct: float = 0.0004):
        self.usdt = float(starting_usdt)
        self.base_by_symbol: Dict[str, float] = {}
        self.last_price: Dict[str, float] = {}
        self.fee_pct = float(fee_pct)
        self._orders: Dict[str, OrderResult] = {}

    def update_last_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = float(price)

    def fetch_klines(
        self, *, symbol: str, interval_minutes: int, start_ms: Optional[int], limit: int = 1000
    ) -> List[Kline]:
        # Paper mode does not pull from the network. Data-syncer should be disabled or replaced by seed scripts.
        return []

    def place_market_order(self, *, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side={side}")

        px = float(self.last_price.get(symbol, 0.0))
        qty_f = max(0.0, float(qty))
        fee = qty_f * px * self.fee_pct

        base_qty = float(self.base_by_symbol.get(symbol, 0.0))
        if side_u == "BUY":
            cost = qty_f * px + fee
            self.usdt -= cost
            base_qty += qty_f
        else:
            # sell up to base_qty
            qty_f = min(qty_f, base_qty)
            proceeds = qty_f * px - fee
            base_qty -= qty_f
            self.usdt += max(0.0, proceeds)

        self.base_by_symbol[symbol] = base_qty

        oid = f"paper_{int(time.time() * 1000)}"
        res = OrderResult(
            exchange_order_id=oid,
            status="FILLED",
            filled_qty=qty_f,
            avg_price=px,
            fee_usdt=fee,
            pnl_usdt=None,
            raw={"usdt": self.usdt, "base_qty": base_qty},
        )
        self._orders[client_order_id] = res
        return res

    def place_stop_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: str,
        reduce_only: bool = True,
    ) -> OrderResult:
        # Accept the order but keep it pending; paper mode strategy will handle stop conditions itself.
        oid = f"paper_stop_{int(time.time() * 1000)}"
        res = OrderResult(
            exchange_order_id=oid,
            status="NEW",
            filled_qty=0.0,
            avg_price=None,
            raw={"symbol": symbol, "side": side, "qty": float(qty), "stop_price": float(stop_price), "reduce_only": reduce_only},
        )
        self._orders[client_order_id] = res
        return res

    def cancel_order(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> bool:
        # Best-effort: mark canceled in local map.
        if client_order_id in self._orders:
            cur = self._orders[client_order_id]
            self._orders[client_order_id] = OrderResult(
                exchange_order_id=cur.exchange_order_id,
                status="CANCELED",
                filled_qty=cur.filled_qty,
                avg_price=cur.avg_price,
                fee_usdt=cur.fee_usdt,
                pnl_usdt=cur.pnl_usdt,
                raw=cur.raw,
            )
        return True

    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        return self._orders.get(
            client_order_id,
            OrderResult(exchange_order_id=exchange_order_id or "", status="NEW", filled_qty=0.0),
        )
