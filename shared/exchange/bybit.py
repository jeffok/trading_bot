from __future__ import annotations

import hashlib
import hmac
import json
import logging
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


_logger = logging.getLogger(__name__)


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
        
        # 清理和验证API密钥（去除前后空格）
        api_key_original = api_key
        api_secret_original = api_secret
        self.api_key = (api_key or "").strip()
        api_secret_cleaned = (api_secret or "").strip()
        self.api_secret = api_secret_cleaned.encode("utf-8") if api_secret_cleaned else b""
        
        # 记录API密钥初始化状态（不泄露敏感信息）
        if self.api_key:
            key_prefix = self.api_key[:4] if len(self.api_key) >= 4 else "****"
            key_length = len(self.api_key)
            _logger.info(
                f"Bybit API key initialized: prefix={key_prefix}***, length={key_length}, "
                f"service={service_name}"
            )
            # 检查是否有空格被清理
            if api_key_original != self.api_key:
                _logger.warning(
                    f"Bybit API key had leading/trailing whitespace removed. "
                    f"Original length={len(api_key_original)}, cleaned length={key_length}"
                )
        else:
            _logger.warning(
                f"Bybit API key is empty. This may cause retCode=10006 errors for signed requests. "
                f"service={service_name}"
            )
        
        # 验证API密钥格式（Bybit API密钥通常是字母数字字符串）
        if self.api_key and not self.api_key.replace("-", "").replace("_", "").isalnum():
            _logger.warning(
                f"Bybit API key contains unusual characters. "
                f"This may cause authentication issues (retCode=10006). "
                f"prefix={self.api_key[:4] if len(self.api_key) >= 4 else '****'}***"
            )
        
        # 验证API密钥长度（Bybit API密钥通常有一定长度）
        if self.api_key and len(self.api_key) < 10:
            _logger.warning(
                f"Bybit API key seems too short (length={len(self.api_key)}). "
                f"This may cause retCode=10006 errors. "
                f"Please verify your BYBIT_API_KEY environment variable."
            )
        
        if not api_secret_cleaned:
            _logger.warning(
                f"Bybit API secret is empty. This will cause authentication failures for signed requests. "
                f"service={service_name}"
            )
        
        self.recv_window = int(recv_window)
        self.leverage = int(leverage)
        self.position_idx = int(position_idx)

        self.limiter = limiter
        self.metrics = metrics
        self.service_name = service_name

        # 优化后的速率限制预算配置（符合Bybit实际限制）
        # 市场数据：保守配置（20个交易对，需要更谨慎）
        # Bybit公开API相对宽松，但考虑到多交易对，设置为2 RPS
        self.limiter.ensure_budget("market_data", 2, 5)      # 2 RPS, 5 burst
        
        # 账户查询：中等频率（Bybit支持50次/秒，但保守设置为3 RPS）
        self.limiter.ensure_budget("account", 3, 6)          # 3 RPS, 6 burst
        
        # 订单操作：区分创建和查询
        # 订单创建：Bybit限制10-20次/秒，保守设置为2 RPS
        self.limiter.ensure_budget("order_create", 2, 4)     # 2 RPS, 4 burst
        # 订单查询：Bybit支持50次/秒，设置为5 RPS
        self.limiter.ensure_budget("order_query", 5, 10)     # 5 RPS, 10 burst
        # 向后兼容：order预算使用order_create
        self.limiter.ensure_budget("order", 2, 4)             # 向后兼容

        self._prepared_symbols: set[str] = set()
        
        # 查询接口缓存（借鉴trading-ci最佳实践）
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl = {
            "wallet_balance": 1.0,      # 1秒缓存
            "position_list": 1.0,       # 1秒缓存
            "order_status": 0.5,        # 0.5秒缓存
        }

    # -------------------------
    # Bybit V5 签名
    # -------------------------
    def _sign(self, payload: str, ts_ms: int) -> str:
        # v5: prehash = timestamp + api_key + recv_window + payload
        pre = f"{ts_ms}{self.api_key}{self.recv_window}{payload}"
        return hmac.new(self.api_secret, pre.encode("utf-8"), hashlib.sha256).hexdigest()

    def _cache_get(self, key: str, ttl_sec: float) -> Optional[Dict[str, Any]]:
        """获取缓存数据（借鉴trading-ci）"""
        try:
            ts, val = self._cache.get(key, (0.0, {}))
            if not val:
                return None
            if (time.time() - float(ts)) <= float(ttl_sec):
                return val
            return None
        except Exception:
            return None

    def _cache_set(self, key: str, val: Dict[str, Any]) -> None:
        """设置缓存数据（借鉴trading-ci）"""
        try:
            self._cache[key] = (time.time(), val)
        except Exception:
            pass

    def _parse_rate_limit_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """解析Bybit速率限制响应头（借鉴trading-ci）
        
        返回：
        - remaining: 剩余请求次数
        - limit: 总限制次数
        - reset_timestamp_ms: 重置时间戳（毫秒）
        """
        result = {
            "remaining": None,
            "limit": None,
            "reset_timestamp_ms": None,
        }
        
        # 转换为小写键的字典
        headers_lower = {k.lower(): v for k, v in headers.items()}
        
        # X-Bapi-Limit-Status: 剩余次数
        for k in ("x-bapi-limit-status", "x-bapi-limit-remaining"):
            if k in headers_lower:
                try:
                    result["remaining"] = int(float(headers_lower[k]))
                    break
                except Exception:
                    pass
        
        # X-Bapi-Limit: 总限制次数
        for k in ("x-bapi-limit",):
            if k in headers_lower:
                try:
                    result["limit"] = int(float(headers_lower[k]))
                    break
                except Exception:
                    pass
        
        # X-Bapi-Limit-Reset-Timestamp: 重置时间戳
        for k in ("x-bapi-limit-reset-timestamp",):
            if k in headers_lower:
                try:
                    n = int(float(headers_lower[k]))
                    # 如果是秒级时间戳，转换为毫秒
                    if n < 10_000_000_000:
                        n = n * 1000
                    result["reset_timestamp_ms"] = n
                    break
                except Exception:
                    pass
        
        return result

    def _apply_rate_limit_headers(self, budget: str, headers: Dict[str, Any]) -> None:
        """应用速率限制响应头，实现自适应调整（借鉴trading-ci）"""
        try:
            rl_info = self._parse_rate_limit_headers(headers)
            remaining = rl_info.get("remaining")
            limit = rl_info.get("limit")
            
            # 如果剩余次数很少，记录警告
            if remaining is not None and remaining < 5:
                _logger.warning(
                    f"Bybit rate limit approaching: remaining={remaining}, "
                    f"limit={limit}, budget={budget}, service={self.service_name}"
                )
            
            # 如果接近限制，可以在这里实现自适应调整
            # 当前limiter已经支持feedback_ok，会自动处理Retry-After
            # 未来可以扩展limiter支持rate_multiplier（类似trading-ci）
            
        except Exception:
            pass

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        signed: bool,
        budget: str,
        max_retries: int = 3,
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

        # 重试循环（借鉴trading-ci）
        last_exception: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=10) as client:
                    if method.upper() == "GET":
                        resp = client.get(url, params=send_params, headers=headers)
                    else:
                        if signed and body_bytes is not None:
                            resp = client.request(method, url, params=send_params, headers=headers, content=body_bytes)
                        else:
                            resp = client.request(method, url, params=send_params, headers=headers, json=json_body)

                # 记录请求详情（用于调试）
                _logger.debug(
                    f"Bybit request: {method} {path} (attempt {attempt}/{max_retries})",
                    extra={
                        "method": method,
                        "path": path,
                        "signed": signed,
                        "status_code": resp.status_code,
                        "budget": budget,
                        "attempt": attempt,
                    }
                )

                if resp.status_code in (429, 418):
                    retry_after = None
                    ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                    if ra:
                        try:
                            retry_after = float(ra)
                        except Exception:
                            retry_after = None
                    decision = self.limiter.feedback_rate_limited(budget, retry_after_seconds=retry_after, status_code=resp.status_code)
                    # 速率限制错误：可重试
                    if attempt < max_retries:
                        sleep_s = decision.get("backoff_seconds", 1.5)
                        time.sleep(sleep_s)
                        continue
                    raise RateLimitError(
                        message=resp.text[:200],
                        retry_after_seconds=decision.get("backoff_seconds"),
                        group=budget,
                        severe=bool(decision.get("severe")),
                    )
                if resp.status_code in (401, 403):
                    error_text = resp.text[:200]
                    _logger.error(
                        f"Bybit authentication error: {error_text}",
                        extra={
                            "status_code": resp.status_code,
                            "path": path,
                            "method": method,
                            "signed": signed,
                        }
                    )
                    raise AuthError(error_text)
                if resp.status_code >= 500:
                    # 服务器错误：可重试
                    error_text = resp.text[:200]
                    _logger.warning(
                        f"Bybit server error: {error_text} (attempt {attempt}/{max_retries})",
                        extra={
                            "status_code": resp.status_code,
                            "path": path,
                            "method": method,
                            "attempt": attempt,
                        }
                    )
                    if attempt < max_retries:
                        time.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
                        continue
                    raise TemporaryError(error_text)
                if resp.status_code >= 400:
                    error_text = resp.text[:200]
                    _logger.warning(
                        f"Bybit client error: {error_text}",
                        extra={
                            "status_code": resp.status_code,
                            "path": path,
                            "method": method,
                            "signed": signed,
                        }
                    )
                    raise ExchangeError(error_text)

                data = resp.json()

                # Bybit V5: retCode != 0 视为业务错误
                if isinstance(data, dict) and data.get("retCode") not in (0, "0", None):
                    ret_code = data.get("retCode")
                    ret_msg = data.get("retMsg", "Unknown error")
                    
                    # 针对retCode=10006：需要区分速率限制和API密钥错误
                    if ret_code == 10006 or str(ret_code) == "10006":
                        ret_msg_lower = ret_msg.lower()
                        # 检查是否是速率限制错误
                        is_rate_limit = any(
                            keyword in ret_msg_lower
                            for keyword in [
                                "rate limit",
                                "too many visits",
                                "exceeded",
                                "too many requests",
                                "frequency limit",
                            ]
                        )
                        
                        if is_rate_limit:
                            # 速率限制错误：可重试
                            _logger.warning(
                                f"Bybit rate limit (retCode=10006): {ret_msg}",
                                extra={
                                    "retCode": ret_code,
                                    "retMsg": ret_msg,
                                    "path": path,
                                    "method": method,
                                    "signed": signed,
                                    "budget": budget,
                                    "attempt": attempt,
                                }
                            )
                            # 使用自适应退避策略
                            decision = self.limiter.feedback_rate_limited(
                                budget, retry_after_seconds=None, status_code=429
                            )
                            if attempt < max_retries:
                                sleep_s = decision.get("backoff_seconds", 1.5)
                                time.sleep(sleep_s)
                                continue
                            raise RateLimitError(
                                message=f"{ret_msg} (retCode={ret_code})",
                                retry_after_seconds=decision.get("backoff_seconds"),
                                group=budget,
                                severe=bool(decision.get("severe")),
                            )
                        else:
                            # API密钥错误：不可重试，立即抛出
                            error_details = [
                                f"Bybit API error: {ret_msg} (retCode={ret_code})",
                                "",
                                "错误10006通常表示API密钥存在问题。请检查以下事项：",
                                "1. 确认BYBIT_API_KEY环境变量已正确设置且不为空",
                                "2. 确认API密钥没有前导或尾随空格（代码已自动清理）",
                                "3. 确认API密钥格式正确（通常为字母数字字符串）",
                                "4. 在Bybit平台上确认API密钥未被禁用或删除",
                                "5. 确认API密钥具有执行所需操作的权限",
                                "6. 检查服务器时间是否与标准时间同步（误差应在1秒内）",
                                "",
                                f"当前API密钥状态: length={len(self.api_key)}, "
                                f"prefix={self.api_key[:4] if len(self.api_key) >= 4 else 'N/A'}***",
                                f"请求路径: {path}, 方法: {method}, 是否需要签名: {signed}",
                            ]
                            error_msg = "\n".join(error_details)
                            _logger.error(
                                f"Bybit retCode=10006 (API key issue): {ret_msg}",
                                extra={
                                    "retCode": ret_code,
                                    "retMsg": ret_msg,
                                    "path": path,
                                    "method": method,
                                    "signed": signed,
                                    "api_key_length": len(self.api_key),
                                    "api_key_prefix": self.api_key[:4] if len(self.api_key) >= 4 else "N/A",
                                }
                            )
                            raise ExchangeError(error_msg)
                    else:
                        # 其他错误代码使用标准错误信息
                        error_msg = f"{ret_msg} (retCode={ret_code})"
                        _logger.warning(
                            f"Bybit API error: {error_msg}",
                            extra={
                                "retCode": ret_code,
                                "retMsg": ret_msg,
                                "path": path,
                                "method": method,
                            }
                        )
                        raise ExchangeError(error_msg)

                # 解析Bybit速率限制响应头并应用自适应调整
                headers_dict = dict(resp.headers)
                self._apply_rate_limit_headers(budget, headers_dict)
                
                self.limiter.feedback_ok(budget, headers=headers_dict)
                return data
                
            except (RateLimitError, TemporaryError) as e:
                # 可重试错误
                last_exception = e
                if attempt < max_retries:
                    if isinstance(e, RateLimitError):
                        sleep_s = e.retry_after_seconds or 1.5
                    else:
                        sleep_s = min(5.0, 0.5 * (2 ** (attempt - 1)))
                    time.sleep(sleep_s)
                    continue
                raise
            except (AuthError, ExchangeError) as e:
                # 不可重试错误，立即抛出
                raise
            except httpx.TimeoutException as e:
                # 超时错误：可重试
                last_exception = TemporaryError(str(e))
                if attempt < max_retries:
                    time.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
                    continue
                raise TemporaryError(str(e)) from e
        
        # 所有重试都失败
        if last_exception:
            raise last_exception
        raise ExchangeError("Request failed after retries")

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

        data_create = self._request("POST", "/v5/order/create", json_body=payload, signed=True, budget="order_create")
        result = (data_create or {}).get("result") or {}
        order_id = str(result.get("orderId", ""))

        # 优化后的订单状态轮询（使用指数退避，减少API请求）
        # 初始延迟：订单创建后需要时间处理（Bybit通常需要100-500ms）
        time.sleep(0.3)
        
        status = "NEW"
        filled_qty = 0.0
        avg_price: Optional[float] = None
        fee_usdt: Optional[float] = None
        pnl_usdt: Optional[float] = None

        end = time.time() + 10.0
        last_status: Optional[OrderResult] = None
        poll_interval = 0.3  # 初始间隔300ms
        max_interval = 1.0   # 最大间隔1秒（指数退避上限）
        
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

            # 如果已成交或取消，立即退出（避免不必要的后续查询）
            if str(status).upper() in ("FILLED", "CANCELED", "CANCELLED", "REJECTED"):
                break
            
            # 指数退避：逐渐增加轮询间隔，减少API请求频率
            time.sleep(poll_interval)
            poll_interval = min(max_interval, poll_interval * 1.5)

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

        # 优化：减少查询窗口，增加查询间隔，限制查询次数
        deadline = time.time() + 15  # 增加到15秒超时
        end_ms = _now_ms()
        start_ms = end_ms - 10 * 60_000  # 减少到10分钟窗口（更精确，减少数据量）
        
        query_count = 0
        max_queries = 5  # 最多查询5次，避免无限循环

        while time.time() < deadline and query_count < max_queries:
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
                query_count += 1
            except ExchangeError:
                data = None
                query_count += 1

            if not data:
                time.sleep(0.5)  # 从0.3增加到0.5秒，减少API请求频率
                continue

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

            time.sleep(0.5)  # 从0.3增加到0.5秒，减少API请求频率

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

        data = self._request("POST", "/v5/order/create", json_body=payload, signed=True, budget="order_create")
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
            self._request("POST", "/v5/order/cancel", json_body=payload, signed=True, budget="order_create")
            return True
        except Exception:
            return False


    def get_order_status(self, *, symbol: str, client_order_id: str, exchange_order_id: Optional[str]) -> OrderResult:
        """查询订单状态（Bybit V5）- 带缓存优化。

        规则：
        - 优先 realtime（近期/活动订单）
        - realtime 查不到时 fallback history（已归档订单）
        - 使用缓存减少API请求（借鉴trading-ci）
        """
        # 构建缓存键
        cache_key = f"order_status:{symbol}:{client_order_id}:{exchange_order_id or ''}"
        ttl = self._cache_ttl.get("order_status", 0.5)
        
        # 尝试从缓存获取
        cached = self._cache_get(cache_key, ttl)
        if cached is not None:
            # 从缓存数据重建OrderResult
            try:
                return OrderResult(
                    exchange_order_id=cached.get("exchange_order_id"),
                    status=cached.get("status"),
                    filled_qty=cached.get("filled_qty", 0.0),
                    avg_price=cached.get("avg_price"),
                    fee_usdt=cached.get("fee_usdt"),
                    pnl_usdt=cached.get("pnl_usdt"),
                    raw=cached.get("raw", {}),
                )
            except Exception:
                # 缓存数据损坏，继续查询
                pass
        
        params: Dict[str, Any] = {"category": "linear", "symbol": symbol}
        if exchange_order_id:
            params["orderId"] = exchange_order_id
        else:
            params["orderLinkId"] = client_order_id

        data = self._request("GET", "/v5/order/realtime", params=params, signed=True, budget="order_query")
        o: Dict[str, Any] = {}
        try:
            o = (((data.get("result") or {}).get("list") or [{}])[0]) or {}
        except Exception:
            o = {}

        if not o or not o.get("orderId"):
            data2 = self._request("GET", "/v5/order/history", params=params, signed=True, budget="order_query")
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
