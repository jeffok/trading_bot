from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx

from .base import ExchangeClient
from .errors import AuthError, ExchangeError, RateLimitError, TemporaryError
from .rate_limiter import AdaptiveRateLimiter
from .types import Kline, OrderResult


def _now_ms() -> int:
    return int(time.time() * 1000)


class BybitV5LinearClient(ExchangeClient):
    """Bybit V5 USDT 合约（linear，逐仓）客户端。

    重要说明（解决 retCode=10004）：
    Bybit v5 的签名要求：签名用的 payload 必须与实际发送的请求内容完全一致。
    - GET：payload=queryString，必须与 URL 上实际 queryString 完全一致（包括参数顺序/编码）
    - POST：payload=jsonBodyString，必须与实际发送 body 字符串完全一致（空格/换行/键顺序都会影响）
    因此：
    - GET：我们用 list[tuple] 生成 queryString（urlencode），并把同样的 list[tuple] 交给 httpx 发送
    - POST：我们先生成紧凑 JSON 字符串用于签名，同时用 content= 原样发送（不用 httpx 的 json= 重新序列化）
    """

    name = "bybit"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        api_secret: str,
        recv_window: int,
        leverage: int,
        position_idx: int,
        limiter: AdaptiveRateLimiter,
        metrics=None,
        service_name: str = "unknown",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8") if api_secret else b""
        self.recv_window = int(recv_window)
        self.leverage = int(leverage)
        self.position_idx = int(position_idx)

        self.limiter = limiter
        self.metrics = metrics
        self.service_name = service_name

        self.limiter.ensure_budget("market_data", 10, 20)
        self.limiter.ensure_budget("account", 5, 10)
        self.limiter.ensure_budget("order", 5, 10)

        self._prepared_symbols: set[str] = set()

    # -------------------------
    # Bybit V5 签名
    # -------------------------
    def _sign(self, payload: str, ts_ms: int) -> str:
        # v5: prehash = timestamp + api_key + recv_window + payload
        pre = f"{ts_ms}{self.api_key}{self.recv_window}{payload}"
        return hmac.new(self.api_secret, pre.encode("utf-8"), hashlib.sha256).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        signed: bool,
        budget: str,
    ) -> Any:
        url = f"{self.base_url}{path}"
        self.limiter.acquire(budget, 1.0)

        params = params or {}
        headers: Dict[str, str] = {"Accept": "application/json"}

        # 为保证 “签名 payload == 实际发送内容”，这里会：
        # - GET：用 items(list[tuple]) 生成 queryString，并用 items 发送
        # - POST：用 body_str 生成签名，同时用 content=body_str 原样发送
        send_params: Any = params
        body_bytes: Optional[bytes] = None
        payload_str: str = ""

        if signed:
            if not self.api_key or not self.api_secret:
                raise AuthError("Missing Bybit API key/secret")

            ts = _now_ms()

            if method.upper() == "GET":
                items = [(k, params[k]) for k in sorted(params.keys())]
                payload_str = urlencode(items, doseq=True)
                send_params = items  # httpx 会按 list[tuple] 的顺序拼 query
            else:
                payload_str = json.dumps(json_body or {}, separators=(",", ":"), ensure_ascii=False)
                body_bytes = payload_str.encode("utf-8")

            headers.update(
                {
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-TIMESTAMP": str(ts),
                    "X-BAPI-RECV-WINDOW": str(self.recv_window),
                    "X-BAPI-SIGN": self._sign(payload_str, ts),
                    "Content-Type": "application/json",
                }
            )

        try:
            with httpx.Client(timeout=10) as client:
                if method.upper() == "GET":
                    resp = client.get(url, params=send_params, headers=headers)
                else:
                    if signed and body_bytes is not None:
                        resp = client.request(method, url, params=send_params, headers=headers, content=body_bytes)
                    else:
                        resp = client.request(method, url, params=send_params, headers=headers, json=json_body)

            if resp.status_code in (429, 418):
                retry_after = None
                ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except Exception:
                        retry_after = None
                decision = self.limiter.feedback_rate_limited(budget, retry_after_seconds=retry_after, status_code=resp.status_code)
                raise RateLimitError(
                    message=resp.text[:200],
                    retry_after_seconds=decision.get("backoff_seconds"),
                    group=budget,
                    severe=bool(decision.get("severe")),
                )
            if resp.status_code in (401, 403):
                raise AuthError(resp.text[:200])
            if resp.status_code >= 500:
                raise TemporaryError(resp.text[:200])
            if resp.status_code >= 400:
                raise ExchangeError(resp.text[:200])

            data = resp.json()

            # Bybit V5: retCode != 0 视为业务错误
            if isinstance(data, dict) and data.get("retCode") not in (0, "0", None):
                raise ExchangeError(f"{data.get('retMsg')} (retCode={data.get('retCode')})")

            self.limiter.feedback_ok(budget, headers=dict(resp.headers))
            return data
        except httpx.TimeoutException as e:
            raise TemporaryError(str(e)) from e

    # -------------------------
    # 逐仓 + 杠杆（一次性准备）
    # -------------------------
    def _ensure_isolated_and_leverage(self, symbol: str) -> None:
        if symbol in self._prepared_symbols:
            return

        try:
            self._request(
                "POST",
                "/v5/position/switch-isolated",
                json_body={
                    "category": "linear",
                    "symbol": symbol,
                    "tradeMode": 1,  # 1=isolated
                    "buyLeverage": str(self.leverage),
                    "sellLeverage": str(self.leverage),
                },
                signed=True,
                budget="account",
            )
        except ExchangeError:
            # 不阻塞主流程
            pass

        self._prepared_symbols.add(symbol)

    # -------------------------
    # 行情
    # -------------------------
    def fetch_klines(self, *, symbol: str, interval_minutes: int, start_ms: Optional[int], limit: int = 1000) -> List[Kline]:
        params: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "interval": str(int(interval_minutes)),
            "limit": int(limit),
        }
        if start_ms is not None:
            params["start"] = int(start_ms)

        data = self._request("GET", "/v5/market/kline", params=params, signed=False, budget="market_data")

        rows = (((data or {}).get("result") or {}).get("list") or [])
        out: List[Kline] = []
        for row in rows:
            out.append(
                Kline(
                    open_time_ms=int(row[0]),
                    close_time_ms=int(row[0]) + int(interval_minutes) * 60_000,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        out.sort(key=lambda k: k.open_time_ms)
        return out

    # -------------------------
    # 下单
    # -------------------------
    def place_market_order(self, *, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        """下市价单（Market）。

        返回：
        - exchange_order_id：交易所订单号
        - status：订单状态（尽量返回交易所原始状态字符串）
        - filled_qty / avg_price：尽量从订单回报/查询中获取真实成交数据
        - fee_usdt / pnl_usdt：Bybit 只能在平仓（SELL）时较稳定从 closed-pnl 查询到净盈亏；开仓 BUY 通常仅能拿到手续费（若可获取）
        """
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side={side}")

        self._ensure_isolated_and_leverage(symbol)

        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side_u == "BUY" else "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "orderLinkId": client_order_id,
        }

        # SELL 平仓更安全
        if side_u == "SELL":
            payload["reduceOnly"] = True

        # Hedge 模式可能需要 positionIdx（默认 0=One-way）
        if self.position_idx:
            payload["positionIdx"] = int(self.position_idx)

        data_create = self._request("POST", "/v5/order/create", json_body=payload, signed=True, budget="order")
        result = (data_create or {}).get("result") or {}
        order_id = str(result.get("orderId", ""))

        # 创建后立即轮询状态（短轮询 10s，确保拿到 FILLED/最终态）
        status = "NEW"
        filled_qty = 0.0
        avg_price: Optional[float] = None
        fee_usdt: Optional[float] = None
        pnl_usdt: Optional[float] = None

        end = time.time() + 10.0
        last_status: Optional[OrderResult] = None
        while time.time() < end:
            st = self.get_order_status(symbol=symbol, client_order_id=client_order_id, exchange_order_id=order_id)
            last_status = st
            status = str(st.status or status)
            filled_qty = float(st.filled_qty or 0.0)
            if st.avg_price is not None:
                avg_price = st.avg_price
            # Bybit 订单查询里通常有 cumExecFee
            try:
                if st.raw and isinstance(st.raw, dict):
                    o = (((st.raw.get("result") or {}).get("list") or [{}])[0]) or {}
                    cf = o.get("cumExecFee")
                    if cf not in (None, "", "0", 0):
                        fee_usdt = float(cf)
            except Exception:
                pass

            if str(status).upper() in ("FILLED", "CANCELED", "CANCELLED", "REJECTED"):
                break
            time.sleep(0.2)

        # 平仓 SELL：尽量从 closed-pnl 获取真实净盈亏与手续费（更可靠）
        fee2, pnl2 = self._fetch_closed_pnl(symbol=symbol, order_id=order_id, side=side_u)
        if fee2 is not None:
            fee_usdt = fee2
        if pnl2 is not None:
            pnl_usdt = pnl2

        raw_status = last_status.raw if last_status and isinstance(last_status.raw, dict) else {}
        return OrderResult(
            exchange_order_id=order_id,
            status=status,
            filled_qty=filled_qty,
            avg_price=avg_price,
            fee_usdt=fee_usdt,
            pnl_usdt=pnl_usdt,
            raw={"create": data_create, "status": raw_status},
        )

    # -------------------------
    # 平仓结算 -> closedPnl（净值，含手续费影响）
    # -------------------------
    # -------------------------
    def _fetch_closed_pnl(self, *, symbol: str, order_id: str, side: str) -> Tuple[Optional[float], Optional[float]]:
        """返回 (fee_usdt, pnl_usdt)。只在 SELL（平仓）时返回 pnl。"""
        if side != "SELL":
            return None, 0.0

        deadline = time.time() + 12
        end_ms = _now_ms()
        start_ms = end_ms - 15 * 60_000  # 15 分钟窗口

        while time.time() < deadline:
            try:
                data = self._request(
                    "GET",
                    "/v5/position/closed-pnl",
                    params={
                        "category": "linear",
                        "symbol": symbol,
                        "startTime": str(start_ms),
                        "endTime": str(end_ms),
                        "limit": "50",
                    },
                    signed=True,
                    budget="account",
                )
            except ExchangeError:
                data = None

            lst = (((data or {}).get("result") or {}).get("list") or [])
            for row in lst:
                if str(row.get("orderId", "")) == str(order_id):
                    pnl = None
                    try:
                        pnl = round(float(row.get("closedPnl", "0") or 0.0), 2)
                    except Exception:
                        pnl = None

                    fee = None
                    try:
                        of = float(row.get("openFee", "0") or 0.0)
                        cf = float(row.get("closeFee", "0") or 0.0)
                        fee = round(abs(of) + abs(cf), 2)
                    except Exception:
                        fee = None

                    return fee, pnl

            time.sleep(0.3)

        return None, None

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
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side={side}")

        self._ensure_isolated_and_leverage(symbol)

        trigger_direction = 2 if side_u == "SELL" else 1  # 2: falls to trigger, 1: rises to trigger

        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side_u == "BUY" else "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "orderLinkId": client_order_id,
            "triggerPrice": str(stop_price),
            "triggerDirection": trigger_direction,
            "triggerBy": "LastPrice",
            "timeInForce": "GoodTillCancel",
        }
        if reduce_only:
            payload["reduceOnly"] = True
            payload["closeOnTrigger"] = True

        data = self._request("POST", "/v5/order/create", json_body=payload, signed=True, budget="order")
        result = (data or {}).get("result") or {}
        order_id = str(result.get("orderId", ""))

        return OrderResult(exchange_order_id=order_id, status="NEW", filled_qty=0.0, avg_price=None, raw=data)

    def cancel_order(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> bool:
        payload: Dict[str, Any] = {"category": "linear", "symbol": symbol}
        if client_order_id:
            payload["orderLinkId"] = client_order_id
        elif exchange_order_id:
            payload["orderId"] = exchange_order_id
        try:
            self._request("POST", "/v5/order/cancel", json_body=payload, signed=True, budget="order")
            return True
        except Exception:
            return False


    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        """查询订单状态（Bybit V5）。

        规则：
        - 优先 realtime（近期/活动订单）
        - realtime 查不到时 fallback history（已归档订单）
        """
        params: Dict[str, Any] = {"category": "linear", "symbol": symbol}
        if exchange_order_id:
            params["orderId"] = exchange_order_id
        else:
            params["orderLinkId"] = client_order_id

        data = self._request("GET", "/v5/order/realtime", params=params, signed=True, budget="order")
        o: Dict[str, Any] = {}
        try:
            o = (((data.get("result") or {}).get("list") or [{}])[0]) or {}
        except Exception:
            o = {}

        if not o or not o.get("orderId"):
            data2 = self._request("GET", "/v5/order/history", params=params, signed=True, budget="order")
            try:
                o = (((data2.get("result") or {}).get("list") or [{}])[0]) or {}
                data = data2
            except Exception:
                o = {}

        status = str(o.get("orderStatus") or o.get("order_status") or "UNKNOWN")
        filled_qty = 0.0
        try:
            filled_qty = float(o.get("cumExecQty", 0.0) or 0.0)
        except Exception:
            filled_qty = 0.0

        avg_price: Optional[float] = None
        try:
            ap = o.get("avgPrice")
            if ap not in (None, "", "0", 0):
                avg_price = float(ap)
        except Exception:
            avg_price = None

        # fallback: cumExecValue / cumExecQty
        if avg_price is None and filled_qty > 0:
            try:
                cum_value = float(o.get("cumExecValue", 0) or 0.0)
                avg_price = (cum_value / filled_qty) if cum_value > 0 else None
            except Exception:
                avg_price = None

        return OrderResult(
            exchange_order_id=str(o.get("orderId") or exchange_order_id or ""),
            status=status,
            filled_qty=filled_qty,
            avg_price=avg_price,
            raw=data if isinstance(data, dict) else {"raw": str(data)},
        )
