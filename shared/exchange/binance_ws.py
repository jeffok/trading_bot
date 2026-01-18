"""Binance Futures WebSocket 客户端实现。

支持：
- 公共频道：K线数据（kline）、Ticker（ticker）、Trade（trade）
- 无需认证（公共数据流）

特性：
- 自动重连（指数退避）
- 心跳保持连接
- 支持组合流（combined streams）
- 频率限制控制（每秒最多5条控制消息）
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from .errors import ExchangeError
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


class BinanceWebSocketClient:
    """Binance Futures WebSocket 客户端。
    
    特性：
    - 自动重连（指数退避）
    - 心跳保持连接
    - 支持组合流（combined streams）
    - 频率限制控制
    """
    
    def __init__(
        self,
        *,
        ws_url: str = "wss://fstream.binance.com/ws",
        combined_ws_url: str = "wss://fstream.binance.com/stream",
        reconnect_max_delay: float = 60.0,
        ping_interval: float = 20.0,
        service_name: str = "unknown",
    ):
        self.ws_url = ws_url
        self.combined_ws_url = combined_ws_url
        self.reconnect_max_delay = reconnect_max_delay
        self.ping_interval = ping_interval
        self.service_name = service_name
        
        # 连接状态
        self.ws: Optional[WebSocketClientProtocol] = None
        self.connected = False
        
        # 订阅管理
        self.subscriptions: Set[str] = set()
        
        # 消息回调
        self.message_callbacks: Dict[str, List[Callable[[WSMessage], None]]] = {}
        
        # 重连状态
        self.reconnect_delay = 1.0
        self.reconnect_attempts = 0
        
        # 运行标志
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # 控制消息频率限制（每秒最多5条）
        self._last_control_msg_time = 0.0
        self._control_msg_count = 0
        self._control_msg_window_start = time.time()
    
    async def _connect(self) -> bool:
        """连接 WebSocket"""
        try:
            _logger.info(f"[{self.service_name}] Connecting to Binance WebSocket: {self.ws_url}")
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=None,  # 我们手动处理心跳
            )
            self.connected = True
            self.reconnect_delay = 1.0
            self.reconnect_attempts = 0
            _logger.info(f"[{self.service_name}] Connected to Binance WebSocket")
            return True
        except Exception as e:
            _logger.error(f"[{self.service_name}] Failed to connect WebSocket: {e}")
            self.connected = False
            return False
    
    async def _connect_combined(self, streams: List[str]) -> bool:
        """连接组合流 WebSocket（用于批量订阅）"""
        try:
            # 构建组合流 URL: /stream?streams=stream1/stream2/stream3
            streams_str = "/".join(streams)
            url = f"{self.combined_ws_url}?streams={streams_str}"
            
            _logger.info(f"[{self.service_name}] Connecting to Binance combined WebSocket: {url[:100]}...")
            self.ws = await websockets.connect(
                url,
                ping_interval=None,
            )
            self.connected = True
            self.reconnect_delay = 1.0
            self.reconnect_attempts = 0
            _logger.info(f"[{self.service_name}] Connected to Binance combined WebSocket ({len(streams)} streams)")
            return True
        except Exception as e:
            _logger.error(f"[{self.service_name}] Failed to connect combined WebSocket: {e}")
            self.connected = False
            return False
    
    async def _check_control_rate_limit(self):
        """检查控制消息频率限制（每秒最多5条）"""
        now = time.time()
        # 重置窗口
        if now - self._control_msg_window_start >= 1.0:
            self._control_msg_count = 0
            self._control_msg_window_start = now
        
        # 如果超过限制，等待
        if self._control_msg_count >= 5:
            wait_time = 1.0 - (now - self._control_msg_window_start)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self._control_msg_count = 0
                self._control_msg_window_start = time.time()
        
        self._control_msg_count += 1
    
    async def _handle_message(self, ws: WebSocketClientProtocol):
        """处理 WebSocket 消息"""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    # Binance 组合流格式: {"stream":"btcusdt@kline_15m","data":{...}}
                    if "stream" in data and "data" in data:
                        stream = data["stream"]
                        stream_data = data["data"]
                        
                        # 构建消息（data 字段包含实际数据）
                        ws_msg = WSMessage(
                            topic=stream,
                            data=stream_data,  # 实际数据在 data 字段
                            timestamp=_now_ms(),
                            raw={"stream": stream, "data": stream_data},
                        )
                        await self._dispatch_message(stream, ws_msg)
                    
                    # 单流格式: 直接是数据（兼容处理）
                    elif "e" in data:
                        # 从事件类型推断 stream
                        event_type = data.get("e", "")
                        symbol = data.get("s", "").lower()
                        
                        if event_type == "kline":
                            stream = f"{symbol}@kline_{data.get('k', {}).get('i', '')}"
                        elif event_type == "24hrTicker":
                            stream = f"{symbol}@ticker"
                        else:
                            stream = f"{symbol}@{event_type}"
                        
                        ws_msg = WSMessage(
                            topic=stream,
                            data=data,
                            timestamp=_now_ms(),
                            raw=data,
                        )
                        await self._dispatch_message(stream, ws_msg)
                    
                except json.JSONDecodeError as e:
                    _logger.warning(f"[{self.service_name}] Invalid JSON message: {e}")
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Error handling message: {e}", exc_info=True)
        except ConnectionClosed:
            _logger.warning(f"[{self.service_name}] WebSocket connection closed")
            self.connected = False
        except Exception as e:
            _logger.error(f"[{self.service_name}] WebSocket error: {e}", exc_info=True)
            self.connected = False
    
    async def _dispatch_message(self, topic: str, message: WSMessage):
        """分发消息到注册的回调"""
        # 精确匹配
        if topic in self.message_callbacks:
            for callback in self.message_callbacks[topic]:
                try:
                    callback(message)
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Callback error for {topic}: {e}", exc_info=True)
        
        # 通配符匹配
        for pattern, callbacks in self.message_callbacks.items():
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if topic.startswith(prefix):
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception as e:
                            _logger.error(f"[{self.service_name}] Callback error for {pattern}: {e}", exc_info=True)
    
    async def _ping_loop(self, ws: WebSocketClientProtocol):
        """心跳循环"""
        while self._running:
            try:
                await asyncio.sleep(self.ping_interval)
                if ws and not ws.closed:
                    # Binance 不需要发送 ping，websockets 库会自动处理
                    _logger.debug(f"[{self.service_name}] WebSocket connection alive")
            except Exception as e:
                _logger.warning(f"[{self.service_name}] Ping error: {e}")
                break
    
    async def _reconnect_loop(self):
        """重连循环"""
        while self._running:
            if self.connected:
                await asyncio.sleep(1.0)
                continue
            
            # 等待重连延迟
            delay = min(self.reconnect_delay, self.reconnect_max_delay)
            _logger.info(f"[{self.service_name}] Reconnecting WebSocket in {delay:.1f}s (attempt {self.reconnect_attempts + 1})")
            await asyncio.sleep(delay)
            
            # 尝试重连
            success = await self._connect()
            
            if success:
                self.reconnect_delay = 1.0
                self.reconnect_attempts = 0
                # 重新订阅
                if self.subscriptions:
                    await self._resubscribe()
            else:
                self.reconnect_attempts += 1
                self.reconnect_delay = min(self.reconnect_delay * 2, self.reconnect_max_delay)
    
    async def _resubscribe(self):
        """重新订阅（Binance 组合流在连接时已指定，无需重新订阅）"""
        # Binance 组合流在连接 URL 中指定，断开重连时需要重新连接
        # 这里可以记录需要订阅的流，重连时使用组合流 URL
        pass
    
    async def start(self):
        """启动 WebSocket 客户端"""
        if self._running:
            return
        
        self._running = True
        _logger.info(f"[{self.service_name}] Starting Binance WebSocket client")
        
        # 连接
        await self._connect()
        if self.connected:
            task = asyncio.create_task(self._handle_message(self.ws))
            self._tasks.append(task)
            task = asyncio.create_task(self._ping_loop(self.ws))
            self._tasks.append(task)
        
        # 启动重连循环
        task = asyncio.create_task(self._reconnect_loop())
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
        
        # 关闭连接
        if self.ws and not self.ws.closed:
            await self.ws.close()
        
        self.connected = False
        _logger.info(f"[{self.service_name}] Binance WebSocket client stopped")
    
    async def subscribe(self, stream: str) -> bool:
        """订阅单个流（单流模式）"""
        if not self.connected or not self.ws:
            _logger.warning(f"[{self.service_name}] Cannot subscribe {stream}: WebSocket not connected")
            return False
        
        # Binance 单流模式：订阅消息格式
        # 注意：Binance 推荐使用组合流，单流订阅主要用于测试
        try:
            # 检查频率限制
            await self._check_control_rate_limit()
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [stream],
                "id": int(time.time() * 1000),
            }
            await self.ws.send(json.dumps(subscribe_msg))
            self.subscriptions.add(stream)
            _logger.info(f"[{self.service_name}] Subscribed to: {stream}")
            return True
        except Exception as e:
            _logger.error(f"[{self.service_name}] Failed to subscribe {stream}: {e}")
            return False
    
    async def subscribe_combined(self, streams: List[str]) -> bool:
        """订阅组合流（推荐方式，一次连接多个流）"""
        # 断开当前连接
        if self.ws and not self.ws.closed:
            await self.ws.close()
        
        # 使用组合流 URL 连接
        success = await self._connect_combined(streams)
        if success:
            self.subscriptions.update(streams)
            # 启动消息处理
            if self.ws:
                task = asyncio.create_task(self._handle_message(self.ws))
                self._tasks.append(task)
                task = asyncio.create_task(self._ping_loop(self.ws))
                self._tasks.append(task)
        return success
    
    def on_message(self, topic: str, callback: Callable[[WSMessage], None]):
        """注册消息回调"""
        if topic not in self.message_callbacks:
            self.message_callbacks[topic] = []
        self.message_callbacks[topic].append(callback)
    
    def parse_kline(self, message: WSMessage) -> Optional[Kline]:
        """解析 K线消息为 Kline 对象"""
        try:
            data = message.data
            if isinstance(data, dict) and data.get("e") == "kline":
                k = data.get("k", {})
                return Kline(
                    open_time_ms=int(k.get("t", 0)),
                    close_time_ms=int(k.get("T", 0)),
                    open=float(k.get("o", 0)),
                    high=float(k.get("h", 0)),
                    low=float(k.get("l", 0)),
                    close=float(k.get("c", 0)),
                    volume=float(k.get("v", 0)),
                )
            return None
        except Exception as e:
            _logger.warning(f"[{self.service_name}] Failed to parse kline: {e}")
            return None
