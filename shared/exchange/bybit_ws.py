"""Bybit V5 WebSocket 客户端实现。

支持：
- 公共频道：K线数据（kline）、订单簿（orderbook）、ticker等
- 私有频道：订单状态、持仓、账户余额（需要认证）
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlencode

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from .errors import AuthError, ExchangeError
from .types import Kline

_logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class WSMessage:
    """WebSocket 消息封装"""
    topic: str
    data: Dict[str, Any]
    timestamp: int
    raw: Dict[str, Any]


class BybitWebSocketClient:
    """Bybit V5 WebSocket 客户端。

    特性：
    - 自动重连（指数退避）
    - 心跳保持连接
    - 消息队列缓冲
    - 支持公共和私有频道
    """

    def __init__(
        self,
        *,
        public_ws_url: str = "wss://stream.bybit.com/v5/public/linear",
        private_ws_url: str = "wss://stream.bybit.com/v5/private",
        api_key: str = "",
        api_secret: str = "",
        recv_window: int = 5000,
        reconnect_max_delay: float = 60.0,
        ping_interval: float = 20.0,
        service_name: str = "unknown",
    ):
        self.public_ws_url = public_ws_url
        self.private_ws_url = private_ws_url
        self.api_key = (api_key or "").strip()
        api_secret_cleaned = (api_secret or "").strip()
        self.api_secret = api_secret_cleaned.encode("utf-8") if api_secret_cleaned else b""
        self.recv_window = recv_window
        self.reconnect_max_delay = reconnect_max_delay
        self.ping_interval = ping_interval
        self.service_name = service_name

        # 连接状态
        self.public_ws: Optional[WebSocketClientProtocol] = None
        self.private_ws: Optional[WebSocketClientProtocol] = None
        self.public_connected = False
        self.private_connected = False

        # 订阅管理
        self.public_subscriptions: Set[str] = set()
        self.private_subscriptions: Set[str] = set()

        # 消息回调
        self.message_callbacks: Dict[str, List[Callable[[WSMessage], None]]] = {}

        # 重连状态
        self.reconnect_delay = 1.0
        self.reconnect_attempts = 0

        # 运行标志
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def _sign(self, expires: int) -> str:
        """生成 WebSocket 认证签名"""
        if not self.api_key or not self.api_secret:
            raise AuthError("Missing Bybit API key/secret for WebSocket authentication")
        
        # Bybit V5 WebSocket 签名：api_key + expires + signature
        # signature = HMAC_SHA256(api_secret, "GET/realtime{expires}")
        message = f"GET/realtime{expires}"
        signature = hmac.new(self.api_secret, message.encode("utf-8"), hashlib.sha256).hexdigest()
        return signature

    async def _connect_public(self) -> bool:
        """连接公共 WebSocket"""
        try:
            _logger.info(f"[{self.service_name}] Connecting to Bybit public WebSocket: {self.public_ws_url}")
            self.public_ws = await websockets.connect(
                self.public_ws_url,
                ping_interval=None,  # 我们手动处理心跳
            )
            self.public_connected = True
            self.reconnect_delay = 1.0
            self.reconnect_attempts = 0
            _logger.info(f"[{self.service_name}] Connected to Bybit public WebSocket")
            return True
        except Exception as e:
            _logger.error(f"[{self.service_name}] Failed to connect public WebSocket: {e}")
            self.public_connected = False
            return False

    async def _connect_private(self) -> bool:
        """连接私有 WebSocket（需要认证）"""
        if not self.api_key or not self.api_secret:
            _logger.warning(f"[{self.service_name}] Skipping private WebSocket (no API key/secret)")
            return False

        try:
            _logger.info(f"[{self.service_name}] Connecting to Bybit private WebSocket: {self.private_ws_url}")
            expires = int(time.time() * 1000) + self.recv_window
            signature = self._sign(expires)

            # 构建认证参数
            auth_params = {
                "op": "auth",
                "args": [self.api_key, str(expires), signature],
            }

            self.private_ws = await websockets.connect(
                self.private_ws_url,
                ping_interval=None,
            )
            
            # 发送认证
            await self.private_ws.send(json.dumps(auth_params))
            
            # 等待认证响应
            response = await asyncio.wait_for(self.private_ws.recv(), timeout=5.0)
            resp_data = json.loads(response)
            
            if resp_data.get("success") is True:
                self.private_connected = True
                self.reconnect_delay = 1.0
                self.reconnect_attempts = 0
                _logger.info(f"[{self.service_name}] Authenticated to Bybit private WebSocket")
                
                # 重新订阅之前的频道
                if self.private_subscriptions:
                    await self._resubscribe_private()
                
                return True
            else:
                error_msg = resp_data.get("ret_msg", "Authentication failed")
                _logger.error(f"[{self.service_name}] Private WebSocket auth failed: {error_msg}")
                await self.private_ws.close()
                self.private_ws = None
                return False
        except Exception as e:
            _logger.error(f"[{self.service_name}] Failed to connect private WebSocket: {e}")
            self.private_connected = False
            if self.private_ws:
                try:
                    await self.private_ws.close()
                except Exception:
                    pass
                self.private_ws = None
            return False

    async def _resubscribe_private(self):
        """重新订阅私有频道"""
        if not self.private_subscriptions:
            return
        
        for topic in self.private_subscriptions:
            try:
                subscribe_msg = {"op": "subscribe", "args": [topic]}
                await self.private_ws.send(json.dumps(subscribe_msg))
                _logger.debug(f"[{self.service_name}] Resubscribed to private topic: {topic}")
            except Exception as e:
                _logger.warning(f"[{self.service_name}] Failed to resubscribe {topic}: {e}")

    async def _handle_message(self, ws: WebSocketClientProtocol, is_private: bool):
        """处理 WebSocket 消息"""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    # 处理订阅确认
                    if data.get("op") == "subscribe":
                        if data.get("success") is True:
                            topic = data.get("args", [""])[0] if data.get("args") else ""
                            if is_private:
                                self.private_subscriptions.add(topic)
                            else:
                                self.public_subscriptions.add(topic)
                            _logger.debug(f"[{self.service_name}] Subscribed to: {topic}")
                        else:
                            _logger.warning(f"[{self.service_name}] Subscribe failed: {data.get('ret_msg')}")
                        continue
                    
                    # 处理数据推送
                    if "topic" in data and "data" in data:
                        topic = data["topic"]
                        ws_msg = WSMessage(
                            topic=topic,
                            data=data.get("data", {}),
                            timestamp=_now_ms(),
                            raw=data,
                        )
                        await self._dispatch_message(topic, ws_msg)
                    
                    # 处理心跳响应
                    if data.get("op") == "pong":
                        continue
                        
                except json.JSONDecodeError as e:
                    _logger.warning(f"[{self.service_name}] Invalid JSON message: {e}")
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Error handling message: {e}", exc_info=True)
        except ConnectionClosed:
            _logger.warning(f"[{self.service_name}] WebSocket connection closed")
            if is_private:
                self.private_connected = False
            else:
                self.public_connected = False
        except Exception as e:
            _logger.error(f"[{self.service_name}] WebSocket error: {e}", exc_info=True)
            if is_private:
                self.private_connected = False
            else:
                self.public_connected = False

    async def _dispatch_message(self, topic: str, message: WSMessage):
        """分发消息到注册的回调"""
        # 精确匹配
        if topic in self.message_callbacks:
            for callback in self.message_callbacks[topic]:
                try:
                    callback(message)
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Callback error for {topic}: {e}", exc_info=True)
        
        # 通配符匹配（例如 "kline.*" 匹配所有 kline 主题）
        for pattern, callbacks in self.message_callbacks.items():
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if topic.startswith(prefix):
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception as e:
                            _logger.error(f"[{self.service_name}] Callback error for {pattern}: {e}", exc_info=True)

    async def _ping_loop(self, ws: WebSocketClientProtocol, is_private: bool):
        """心跳循环"""
        while self._running:
            try:
                await asyncio.sleep(self.ping_interval)
                if ws:
                    # 检查连接是否关闭（兼容不同版本的 websockets 库）
                    is_closed = False
                    try:
                        is_closed = ws.closed
                    except AttributeError:
                        # websockets 14.x 可能使用不同的属性
                        try:
                            is_closed = ws.close_code is not None
                        except AttributeError:
                            pass
                    
                    if not is_closed:
                        ping_msg = {"op": "ping"}
                        await ws.send(json.dumps(ping_msg))
                        _logger.debug(f"[{self.service_name}] Sent ping ({'private' if is_private else 'public'})")
            except Exception as e:
                _logger.warning(f"[{self.service_name}] Ping error: {e}")
                break

    async def _reconnect_loop(self, is_private: bool):
        """重连循环"""
        # 私有频道重连循环：只在有 API key 时运行
        if is_private and (not self.api_key or not self.api_secret):
            return
        
        while self._running:
            if is_private:
                if self.private_connected:
                    await asyncio.sleep(1.0)
                    continue
            else:
                if self.public_connected:
                    await asyncio.sleep(1.0)
                    continue

            # 等待重连延迟
            delay = min(self.reconnect_delay, self.reconnect_max_delay)
            _logger.info(f"[{self.service_name}] Reconnecting {'private' if is_private else 'public'} WebSocket in {delay:.1f}s (attempt {self.reconnect_attempts + 1})")
            await asyncio.sleep(delay)

            # 尝试重连
            if is_private:
                success = await self._connect_private()
            else:
                success = await self._connect_public()

            if success:
                self.reconnect_delay = 1.0
                self.reconnect_attempts = 0
            else:
                self.reconnect_attempts += 1
                self.reconnect_delay = min(self.reconnect_delay * 2, self.reconnect_max_delay)

    async def start(self):
        """启动 WebSocket 客户端"""
        if self._running:
            return
        
        self._running = True
        _logger.info(f"[{self.service_name}] Starting Bybit WebSocket client")

        # 连接公共频道（公共接口，无需认证）
        await self._connect_public()
        if self.public_connected:
            task = asyncio.create_task(self._handle_message(self.public_ws, is_private=False))
            self._tasks.append(task)
            task = asyncio.create_task(self._ping_loop(self.public_ws, is_private=False))
            self._tasks.append(task)
        
        # 连接私有频道（仅在有 API key 时连接）
        if self.api_key and self.api_secret:
            await self._connect_private()
            if self.private_connected:
                task = asyncio.create_task(self._handle_message(self.private_ws, is_private=True))
                self._tasks.append(task)
                task = asyncio.create_task(self._ping_loop(self.private_ws, is_private=True))
                self._tasks.append(task)
        
        # 启动重连循环
        task = asyncio.create_task(self._reconnect_loop(is_private=False))
        self._tasks.append(task)
        # 只在有 API key 时启动私有频道重连循环
        if self.api_key and self.api_secret:
            task = asyncio.create_task(self._reconnect_loop(is_private=True))
            self._tasks.append(task)

    async def stop(self):
        """停止 WebSocket 客户端"""
        self._running = False
        
        # 取消所有任务
        for task in self._tasks:
            task.cancel()
        
        # 等待任务完成
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # 关闭连接（兼容不同版本的 websockets 库）
        if self.public_ws:
            try:
                is_closed = getattr(self.public_ws, 'closed', None)
                if is_closed is None:
                    is_closed = getattr(self.public_ws, 'close_code', None) is not None
                if not is_closed:
                    await self.public_ws.close()
            except Exception:
                pass
        if self.private_ws:
            try:
                is_closed = getattr(self.private_ws, 'closed', None)
                if is_closed is None:
                    is_closed = getattr(self.private_ws, 'close_code', None) is not None
                if not is_closed:
                    await self.private_ws.close()
            except Exception:
                pass
        
        self.public_connected = False
        self.private_connected = False
        _logger.info(f"[{self.service_name}] Bybit WebSocket client stopped")

    async def subscribe_public(self, topic: str) -> bool:
        """订阅公共频道"""
        if not self.public_connected or not self.public_ws:
            _logger.warning(f"[{self.service_name}] Cannot subscribe {topic}: public WebSocket not connected")
            return False

        try:
            subscribe_msg = {"op": "subscribe", "args": [topic]}
            await self.public_ws.send(json.dumps(subscribe_msg))
            self.public_subscriptions.add(topic)
            _logger.info(f"[{self.service_name}] Subscribed to public topic: {topic}")
            return True
        except Exception as e:
            _logger.error(f"[{self.service_name}] Failed to subscribe {topic}: {e}")
            return False

    async def subscribe_private(self, topic: str) -> bool:
        """订阅私有频道"""
        if not self.private_connected or not self.private_ws:
            _logger.warning(f"[{self.service_name}] Cannot subscribe {topic}: private WebSocket not connected")
            return False

        try:
            subscribe_msg = {"op": "subscribe", "args": [topic]}
            await self.private_ws.send(json.dumps(subscribe_msg))
            self.private_subscriptions.add(topic)
            _logger.info(f"[{self.service_name}] Subscribed to private topic: {topic}")
            return True
        except Exception as e:
            _logger.error(f"[{self.service_name}] Failed to subscribe {topic}: {e}")
            return False

    def on_message(self, topic: str, callback: Callable[[WSMessage], None]):
        """注册消息回调"""
        if topic not in self.message_callbacks:
            self.message_callbacks[topic] = []
        self.message_callbacks[topic].append(callback)

    def parse_kline(self, message: WSMessage) -> Optional[Kline]:
        """解析 K线消息为 Kline 对象"""
        try:
            data = message.data
            if isinstance(data, list) and len(data) > 0:
                k = data[0]  # Bybit 返回数组，取第一个
            elif isinstance(data, dict):
                k = data
            else:
                return None

            return Kline(
                open_time_ms=int(k.get("start", 0)),
                close_time_ms=int(k.get("end", 0)),
                open=float(k.get("open", 0)),
                high=float(k.get("high", 0)),
                low=float(k.get("low", 0)),
                close=float(k.get("close", 0)),
                volume=float(k.get("volume", 0)),
            )
        except Exception as e:
            _logger.warning(f"[{self.service_name}] Failed to parse kline: {e}")
            return None
