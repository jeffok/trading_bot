
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional

import httpx

from .base import ExchangeClient
from .errors import AuthError, ExchangeError, RateLimitError, TemporaryError
from .rate_limiter import AdaptiveRateLimiter
from .types import Kline, OrderResult

class BinanceSpotClient(ExchangeClient):
    name = "binance"

    def __init__(self, *, base_url: str, api_key: str, api_secret: str, recv_window: int,
                 limiter: AdaptiveRateLimiter, metrics=None, service_name: str = "unknown"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8") if api_secret else b""
        self.recv_window = recv_window
        self.limiter = limiter
        self.metrics = metrics
        self.service_name = service_name

        self.limiter.ensure_budget("binance_public", 10, 10)
        self.limiter.ensure_budget("binance_private", 5, 5)

    def _sign(self, qs: str) -> str:
        return hmac.new(self.api_secret, qs.encode("utf-8"), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, *, params: Dict[str, Any], signed: bool, budget: str) -> Any:
        url = f"{self.base_url}{path}"
        self.limiter.acquire(budget, 1.0)

        headers = {"Accept": "application/json"}
        if signed:
            if not self.api_key or not self.api_secret:
                raise AuthError("Missing Binance API key/secret")
            headers["X-MBX-APIKEY"] = self.api_key
            params = dict(params)
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window
            qs = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
            params["signature"] = self._sign(qs)

        start = time.time()
        status_label = "ok"
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.request(method, url, params=params, headers=headers)
            if resp.status_code in (429, 418):
                self.limiter.backoff(budget, 2.0)
                raise RateLimitError(resp.text[:200])
            if resp.status_code in (401, 403):
                raise AuthError(resp.text[:200])
            if resp.status_code >= 500:
                raise TemporaryError(resp.text[:200])
            if resp.status_code >= 400:
                raise ExchangeError(resp.text[:200])
            return resp.json()
        finally:
            elapsed = time.time() - start
            if self.metrics:
                self.metrics.exchange_requests_total.labels(self.service_name, self.name, path, status_label).inc()
                self.metrics.exchange_latency_seconds.labels(self.service_name, self.name, path).observe(elapsed)

    def fetch_klines(self, *, symbol: str, interval_minutes: int, start_ms: Optional[int], limit: int = 1000) -> List[Kline]:
        params: Dict[str, Any] = {"symbol": symbol, "interval": f"{interval_minutes}m", "limit": min(limit, 1500)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        data = self._request("GET", "/api/v3/klines", params=params, signed=False, budget="binance_public")
        out: List[Kline] = []
        for row in data:
            out.append(Kline(
                open_time_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time_ms=int(row[6]),
            ))
        return out

    def place_market_order(self, *, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty, "newClientOrderId": client_order_id}
        data = self._request("POST", "/api/v3/order", params=params, signed=True, budget="binance_private")
        return OrderResult(
            exchange_order_id=str(data.get("orderId", "")),
            status=str(data.get("status", "UNKNOWN")),
            filled_qty=float(data.get("executedQty", 0.0)),
            avg_price=None,
            raw=data,
        )

    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        params: Dict[str, Any] = {"symbol": symbol}
        if exchange_order_id:
            params["orderId"] = exchange_order_id
        else:
            params["origClientOrderId"] = client_order_id
        data = self._request("GET", "/api/v3/order", params=params, signed=True, budget="binance_private")
        return OrderResult(
            exchange_order_id=str(data.get("orderId", "")),
            status=str(data.get("status", "UNKNOWN")),
            filled_qty=float(data.get("executedQty", 0.0)),
            avg_price=None,
            raw=data,
        )
