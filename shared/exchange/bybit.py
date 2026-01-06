
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional

import httpx

from .base import ExchangeClient
from .errors import AuthError, ExchangeError, RateLimitError, TemporaryError
from .rate_limiter import AdaptiveRateLimiter
from .types import Kline, OrderResult

class BybitV5SpotClient(ExchangeClient):
    name = "bybit"

    def __init__(self, *, base_url: str, api_key: str, api_secret: str, recv_window: int,
                 limiter: AdaptiveRateLimiter, metrics=None, service_name: str = "unknown"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8") if api_secret else b""
        self.recv_window = recv_window
        self.limiter = limiter
        self.metrics = metrics
        self.service_name = service_name

        self.limiter.ensure_budget("bybit_public", 10, 10)
        self.limiter.ensure_budget("bybit_private", 5, 5)

    def _hmac(self, text: str) -> str:
        return hmac.new(self.api_secret, text.encode("utf-8"), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                 json_body: Optional[Dict[str, Any]] = None, signed: bool, budget: str) -> Any:
        url = f"{self.base_url}{path}"
        self.limiter.acquire(budget, 1.0)

        headers = {"Accept": "application/json"}
        ts = str(int(time.time() * 1000))
        recv = str(self.recv_window)

        if signed:
            if not self.api_key or not self.api_secret:
                raise AuthError("Missing Bybit API key/secret")
            headers["X-BAPI-API-KEY"] = self.api_key
            headers["X-BAPI-TIMESTAMP"] = ts
            headers["X-BAPI-RECV-WINDOW"] = recv
            headers["Content-Type"] = "application/json"

            if method.upper() == "GET":
                params = params or {}
                qs = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
                sign_text = f"{ts}{self.api_key}{recv}{qs}"
            else:
                body_str = json.dumps(json_body or {}, separators=(",", ":"), ensure_ascii=False)
                sign_text = f"{ts}{self.api_key}{recv}{body_str}"

            headers["X-BAPI-SIGN"] = self._hmac(sign_text)

        start = time.time()
        status_label = "ok"
        try:
            with httpx.Client(timeout=10) as client:
                if method.upper() == "GET":
                    resp = client.get(url, params=params, headers=headers)
                else:
                    resp = client.request(method, url, params=params, json=json_body, headers=headers)

            if resp.status_code == 429:
                self.limiter.backoff(budget, 2.0)
                raise RateLimitError(resp.text[:200])
            if resp.status_code in (401, 403):
                raise AuthError(resp.text[:200])
            if resp.status_code >= 500:
                raise TemporaryError(resp.text[:200])
            if resp.status_code >= 400:
                raise ExchangeError(resp.text[:200])

            data = resp.json()
            if isinstance(data, dict) and data.get("retCode") not in (0, None):
                code = data.get("retCode")
                msg = data.get("retMsg", "")
                if code in (10006,):
                    self.limiter.backoff(budget, 2.0)
                    raise RateLimitError(f"{code} {msg}")
                raise ExchangeError(f"{code} {msg}")
            return data
        finally:
            elapsed = time.time() - start
            if self.metrics:
                self.metrics.exchange_requests_total.labels(self.service_name, self.name, path, status_label).inc()
                self.metrics.exchange_latency_seconds.labels(self.service_name, self.name, path).observe(elapsed)

    def fetch_klines(self, *, symbol: str, interval_minutes: int, start_ms: Optional[int], limit: int = 1000) -> List[Kline]:
        params: Dict[str, Any] = {"category": "spot", "symbol": symbol, "interval": str(interval_minutes), "limit": str(min(limit, 1000))}
        if start_ms is not None:
            params["start"] = int(start_ms)

        data = self._request("GET", "/v5/market/kline", params=params, signed=False, budget="bybit_public")
        rows = (data.get("result") or {}).get("list") or []
        out: List[Kline] = []
        for r in rows:
            out.append(Kline(
                open_time_ms=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                close_time_ms=int(r[0]) + interval_minutes * 60_000 - 1,
            ))
        out.sort(key=lambda k: k.open_time_ms)
        return out

    def place_market_order(self, *, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        body = {"category": "spot", "symbol": symbol, "side": "Buy" if side.upper() == "BUY" else "Sell",
                "orderType": "Market", "qty": str(qty), "orderLinkId": client_order_id}
        data = self._request("POST", "/v5/order/create", json_body=body, signed=True, budget="bybit_private")
        res = data.get("result") or {}
        return OrderResult(exchange_order_id=str(res.get("orderId", "")), status="CREATED", filled_qty=0.0, avg_price=None, raw=data)

    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        params: Dict[str, Any] = {"category": "spot", "symbol": symbol}
        if exchange_order_id:
            params["orderId"] = exchange_order_id
        else:
            params["orderLinkId"] = client_order_id

        data = self._request("GET", "/v5/order/realtime", params=params, signed=True, budget="bybit_private")
        lst = (data.get("result") or {}).get("list") or []
        if not lst:
            return OrderResult(exchange_order_id=exchange_order_id or "", status="UNKNOWN", filled_qty=0.0, raw=data)
        o = lst[0]
        return OrderResult(
            exchange_order_id=str(o.get("orderId", "")),
            status=str(o.get("orderStatus", "UNKNOWN")),
            filled_qty=float(o.get("cumExecQty", 0.0) or 0.0),
            avg_price=float(o.get("avgPrice")) if o.get("avgPrice") else None,
            raw=o,
        )
