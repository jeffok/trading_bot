from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .base import ExchangeClient
from .errors import AuthError, ExchangeError, RateLimitError, TemporaryError
from .rate_limiter import AdaptiveRateLimiter
from .types import Kline, OrderResult


def _minutes_to_binance_interval(minutes: int) -> str:
    """把分钟转换为 Binance interval 字符串。"""
    mapping = {
        1: "1m",
        3: "3m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "1h",
        120: "2h",
        240: "4h",
        360: "6h",
        480: "8h",
        720: "12h",
        1440: "1d",
    }
    if minutes not in mapping:
        raise ValueError(f"Unsupported interval_minutes={minutes} for Binance")
    return mapping[minutes]


class BinanceUsdtFuturesClient(ExchangeClient):
    """Binance USDT-M Futures（逐仓）客户端。

    重点：
    - 只使用 USDT-M Futures（/fapi/v1/...）
    - 启动时对 symbol 设置：
      1) marginType=ISOLATED
      2) leverage=FUTURES_LEVERAGE
    - SELL（平仓）后，通过 userTrades 获取已结算的 realizedPnl + commission，从而得到真实净盈亏（USDT，保留 2 位小数）
      说明：funding fee 在 Binance 是独立的 income 记录，不一定能严格归集到某一个 orderId。
    """

    name = "binance"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        api_secret: str,
        recv_window: int,
        leverage: int,
        limiter: AdaptiveRateLimiter,
        metrics=None,
        service_name: str = "unknown",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8") if api_secret else b""
        self.recv_window = int(recv_window)
        self.leverage = int(leverage)

        self.limiter = limiter
        self.metrics = metrics
        self.service_name = service_name

        # 预算：public 10 rps / private 5 rps（保守值）
        self.limiter.ensure_budget("market_data", 10, 20)
        self.limiter.ensure_budget("account", 5, 10)
        self.limiter.ensure_budget("order", 5, 10)

        # 记忆：避免每次下单都重复设置
        self._prepared_symbols: set[str] = set()

    # -------------------------
    # HTTP + 签名
    # -------------------------
    def _sign(self, qs: str) -> str:
        return hmac.new(self.api_secret, qs.encode("utf-8"), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, *, params: Dict[str, Any], signed: bool, budget: str) -> Any:
        url = f"{self.base_url}{path}"
        self.limiter.acquire(budget, 1.0)

        headers = {"Accept": "application/json"}
        params2 = dict(params)

        if signed:
            if not self.api_key or not self.api_secret:
                raise AuthError("Missing Binance API key/secret")
            headers["X-MBX-APIKEY"] = self.api_key
            params2["timestamp"] = int(time.time() * 1000)
            params2["recvWindow"] = self.recv_window
            qs = "&".join([f"{k}={params2[k]}" for k in sorted(params2.keys())])
            params2["signature"] = self._sign(qs)

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.request(method, url, params=params2, headers=headers)

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

            self.limiter.feedback_ok(budget, headers=dict(resp.headers))
            return resp.json()
        except httpx.TimeoutException as e:
            raise TemporaryError(str(e)) from e

    # -------------------------
    # 逐仓 + 杠杆（一次性准备）
    # -------------------------
    def _ensure_isolated_and_leverage(self, symbol: str) -> None:
        """确保逐仓 + 杠杆已设置。"""
        if symbol in self._prepared_symbols:
            return

        # 1) 设置逐仓：如果已是 ISOLATED，Binance 可能返回特定错误码/提示，这里忽略即可
        try:
            self._request(
                "POST",
                "/fapi/v1/marginType",
                params={"symbol": symbol, "marginType": "ISOLATED"},
                signed=True,
                budget="account",
            )
        except ExchangeError:
            # 已设置/不支持等情况：不阻塞策略运行
            pass

        # 2) 设置杠杆
        try:
            self._request(
                "POST",
                "/fapi/v1/leverage",
                params={"symbol": symbol, "leverage": str(self.leverage)},
                signed=True,
                budget="account",
            )
        except ExchangeError:
            pass

        self._prepared_symbols.add(symbol)

    # -------------------------
    # 公共行情
    # -------------------------
    def fetch_klines(self, *, symbol: str, interval_minutes: int, start_ms: Optional[int], limit: int = 1000) -> List[Kline]:
        interval = _minutes_to_binance_interval(interval_minutes)
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": int(limit)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)

        data = self._request("GET", "/fapi/v1/klines", params=params, signed=False, budget="market_data")
        out: List[Kline] = []
        for row in data:
            # row layout (futures klines) 与 spot 基本一致
            out.append(
                Kline(
                    open_time_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time_ms=int(row[6]),
                )
            )
        return out

    # -------------------------
    # 下单 / 查询
    # -------------------------
    def place_market_order(self, *, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult:
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side={side}")

        self._ensure_isolated_and_leverage(symbol)

        # reduceOnly：本项目为“单向做多 -> 平仓卖出”，SELL 一律 reduceOnly 更安全
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side_u,
            "type": "MARKET",
            "quantity": str(qty),
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        if side_u == "SELL":
            params["reduceOnly"] = "true"

        data = self._request("POST", "/fapi/v1/order", params=params, signed=True, budget="order")

        order_id = str(data.get("orderId", ""))
        status = str(data.get("status", "UNKNOWN"))
        executed_qty = float(data.get("executedQty", 0.0) or 0.0)

        # avgPrice 有时在 RESULT 中返回字符串
        avg_price = None
        if data.get("avgPrice") not in (None, "", "0", 0):
            try:
                avg_price = float(data["avgPrice"])
            except Exception:
                avg_price = None

        # 确保 FILLED
        if status != "FILLED":
            end = time.time() + 10
            while time.time() < end:
                st = self.get_order_status(symbol=symbol, client_order_id=client_order_id, exchange_order_id=order_id)
                status = st.status
                executed_qty = st.filled_qty
                if status == "FILLED":
                    break
                time.sleep(0.2)

        # 从成交/结算记录拿真实 pnl / fee
        fee_usdt, pnl_usdt = self._fetch_trade_pnl_and_fee(symbol=symbol, order_id=order_id, side=side_u)

        return OrderResult(
            exchange_order_id=order_id,
            status=status,
            filled_qty=executed_qty,
            avg_price=avg_price,
            fee_usdt=fee_usdt,
            pnl_usdt=pnl_usdt,
            raw=data,
        )

    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        params: Dict[str, Any] = {"symbol": symbol}
        if exchange_order_id:
            params["orderId"] = exchange_order_id
        else:
            params["origClientOrderId"] = client_order_id

        data = self._request("GET", "/fapi/v1/order", params=params, signed=True, budget="order")
        return OrderResult(
            exchange_order_id=str(data.get("orderId", "")),
            status=str(data.get("status", "UNKNOWN")),
            filled_qty=float(data.get("executedQty", 0.0) or 0.0),
            avg_price=(float(data.get("avgPrice", 0.0) or 0.0) or None),
            raw=data,
        )

    # -------------------------
    # 成交明细 -> 净 PnL（已扣手续费）
    # -------------------------
    def _fetch_trade_pnl_and_fee(self, *, symbol: str, order_id: str, side: str) -> Tuple[Optional[float], Optional[float]]:
        """返回 (fee_usdt, pnl_usdt)。

        Binance USDT-M Futures 的 userTrades 返回：
        - realizedPnl：该成交产生的已实现盈亏（不含手续费）
        - commission：该成交手续费
        - commissionAsset：通常为 USDT
        所以净盈亏（USDT）≈ sum(realizedPnl) - sum(commission)
        """

        # BUY 开仓：pnl 通常为 0；但我们仍返回手续费，方便告警/统计
        deadline = time.time() + 10

        while time.time() < deadline:
            try:
                trades = self._request(
                    "GET",
                    "/fapi/v1/userTrades",
                    params={"symbol": symbol, "orderId": order_id},
                    signed=True,
                    budget="account",
                )
            except ExchangeError:
                trades = []

            if isinstance(trades, list) and trades:
                realized = 0.0
                fee = 0.0
                fee_asset_ok = True

                for t in trades:
                    # realizedPnl / commission 都是字符串
                    try:
                        realized += float(t.get("realizedPnl", "0") or 0.0)
                    except Exception:
                        pass

                    try:
                        comm = float(t.get("commission", "0") or 0.0)
                    except Exception:
                        comm = 0.0

                    ca = str(t.get("commissionAsset", "") or "")
                    if ca and ca != "USDT":
                        # 如果出现非 USDT 手续费资产（罕见），无法直接换算，先标记为不可用
                        fee_asset_ok = False
                    fee += comm

                fee_out = round(fee, 2) if fee_asset_ok else None

                if side == "SELL":
                    pnl_net = realized - fee
                    pnl_out = round(pnl_net, 2) if fee_asset_ok else None
                else:
                    pnl_out = 0.0

                return fee_out, pnl_out

            time.sleep(0.2)

        # 超时：返回 None（上层告警可展示“未知”）
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

        # Ensure isolated + leverage for safety (idempotent)
        self._ensure_isolated_and_leverage(symbol)

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side_u,
            "type": "STOP_MARKET",
            "stopPrice": str(stop_price),
            "quantity": str(qty),
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        # extra safety flags (best-effort)
        params["priceProtect"] = "TRUE"
        params["workingType"] = "CONTRACT_PRICE"

        data = self._request("POST", "/fapi/v1/order", params=params, signed=True, budget="order")

        order_id = str(data.get("orderId") or "")
        status = str(data.get("status") or "NEW")
        executed_qty = float(data.get("executedQty") or 0.0)
        avg_price = None
        try:
            ap = data.get("avgPrice")
            if ap is not None and str(ap) != "":
                avg_price = float(ap)
        except Exception:
            avg_price = None

        return OrderResult(
            exchange_order_id=order_id,
            status=status,
            filled_qty=executed_qty,
            avg_price=avg_price,
            raw=data,
        )

    def cancel_order(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> bool:
        params: Dict[str, Any] = {"symbol": symbol}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        elif exchange_order_id:
            params["orderId"] = exchange_order_id

        try:
            self._request("DELETE", "/fapi/v1/order", params=params, signed=True, budget="order")
            return True
        except Exception:
            return False
