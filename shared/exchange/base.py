
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from .types import Kline, OrderResult

class ExchangeClient(ABC):
    name: str

    @abstractmethod
    def fetch_klines(self, *, symbol: str, interval_minutes: int, start_ms: Optional[int], limit: int = 1000) -> List[Kline]:
        raise NotImplementedError

    @abstractmethod
    def place_market_order(self, *, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        raise NotImplementedError
