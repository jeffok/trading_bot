"""K线 WebSocket 服务（独立连接）。

专门用于订阅和保存 K线数据到数据库。
价格数据由独立的 PriceWebSocketService 处理。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from shared.domain.time import now_ms
from shared.exchange.types import Kline

_logger = logging.getLogger(__name__)


class KlineWebSocketService:
    """K线 WebSocket 服务（仅订阅 K线流）。
    
    负责：
    - 订阅所有交易对的 K线流
    - 接收已关闭的 K线数据
    - 通过回调保存到数据库
    """
    
    def __init__(
        self,
        exchange: str,
        symbols: List[str],
        interval_minutes: int,
        kline_callback: Callable[[str, Kline], None],
        *,
        service_name: str = "kline-ws",
    ):
        self.exchange = exchange.lower()
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.kline_callback = kline_callback
        self.service_name = service_name
        
        # WebSocket 客户端
        self.ws_client = None
        self._running = False
        self._lock = threading.Lock()
        
        # 订阅管理
        self.subscribed_topics: Set[str] = set()
        
        # 同步状态跟踪
        self.sync_status: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
    def _get_ws_client(self):
        """获取对应交易所的 WebSocket 客户端"""
        if self.exchange == "bybit":
            from shared.exchange.bybit_ws import BybitWebSocketClient
            return BybitWebSocketClient(
                public_ws_url="wss://stream.bybit.com/v5/public/linear",
                service_name=self.service_name,
            )
        elif self.exchange == "binance":
            from shared.exchange.binance_ws import BinanceWebSocketClient
            return BinanceWebSocketClient(
                ws_url="wss://fstream.binance.com/ws",
                service_name=self.service_name,
            )
        else:
            raise ValueError(f"Unsupported exchange for WebSocket: {self.exchange}")
    
    def _minutes_to_bybit_interval(self, minutes: int) -> str:
        """转换分钟数为 Bybit K线间隔格式"""
        if minutes < 60:
            return f"{minutes}"
        elif minutes == 60:
            return "60"
        elif minutes == 240:
            return "240"
        elif minutes == 1440:
            return "D"
        elif minutes == 10080:
            return "W"
        elif minutes == 43200:
            return "M"
        else:
            return str(minutes)
    
    def _minutes_to_binance_interval(self, minutes: int) -> str:
        """转换分钟数为 Binance K线间隔格式"""
        if minutes == 1:
            return "1m"
        elif minutes == 3:
            return "3m"
        elif minutes == 5:
            return "5m"
        elif minutes == 15:
            return "15m"
        elif minutes == 30:
            return "30m"
        elif minutes == 60:
            return "1h"
        elif minutes == 240:
            return "4h"
        elif minutes == 1440:
            return "1d"
        else:
            return f"{minutes}m"
    
    async def _subscribe_kline_streams(self):
        """订阅所有交易对的 K线流"""
        # 检查连接状态
        is_connected = False
        if self.exchange == "bybit":
            is_connected = self.ws_client and hasattr(self.ws_client, 'public_connected') and self.ws_client.public_connected
        elif self.exchange == "binance":
            is_connected = self.ws_client and hasattr(self.ws_client, 'connected') and self.ws_client.connected
        
        if not is_connected:
            _logger.warning(f"[{self.service_name}] Cannot subscribe: WebSocket not connected")
            return
        
        topics = []
        
        # Bybit 订阅格式
        if self.exchange == "bybit":
            interval_str = self._minutes_to_bybit_interval(self.interval_minutes)
            for symbol in self.symbols:
                # K线流: kline.{interval}.{symbol}
                kline_topic = f"kline.{interval_str}.{symbol}"
                topics.append(kline_topic)
        
        # Binance 订阅格式（使用组合流）
        elif self.exchange == "binance":
            interval_str = self._minutes_to_binance_interval(self.interval_minutes)
            streams = []
            for symbol in self.symbols:
                symbol_lower = symbol.lower()
                # K线流: {symbol}@kline_{interval}
                kline_stream = f"{symbol_lower}@kline_{interval_str}"
                streams.append(kline_stream)
            
            # Binance 使用组合流（一次连接订阅所有流）
            if len(streams) <= 1024:
                try:
                    success = await self.ws_client.subscribe_combined(streams)
                    if success:
                        self.subscribed_topics.update(streams)
                        _logger.info(f"[{self.service_name}] Subscribed {len(streams)} kline streams via combined WebSocket")
                        return
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Failed to subscribe combined streams: {e}")
            
            # 降级：单流订阅
            topics = streams
        
        # 批量订阅（控制频率）
        batch_size = 10
        for i in range(0, len(topics), batch_size):
            batch = topics[i:i + batch_size]
            for topic in batch:
                try:
                    if self.exchange == "bybit":
                        await self.ws_client.subscribe_public(topic)
                    elif self.exchange == "binance":
                        await self.ws_client.subscribe(topic)
                    self.subscribed_topics.add(topic)
                    _logger.info(f"[{self.service_name}] Subscribed kline: {topic}")
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Failed to subscribe {topic}: {e}")
            
            if i + batch_size < len(topics):
                await asyncio.sleep(0.2)
        
        _logger.info(f"[{self.service_name}] Subscribed {len(topics)} kline topics for {len(self.symbols)} symbols")
    
    def _on_kline_message(self, message):
        """处理 K线消息"""
        try:
            if self.exchange == "bybit":
                # Bybit V5 K线消息：检查 confirm 字段
                data = message.data if hasattr(message, 'data') else (message.raw.get("data", []) if isinstance(message.raw.get("data"), list) else [])
                
                if isinstance(data, list) and len(data) > 0:
                    kline_data = data[0]
                elif isinstance(data, dict):
                    kline_data = data
                else:
                    return
                
                # 检查 confirms 字段（Bybit V5 中确认的 K线才是已关闭的）
                confirms = kline_data.get("confirm", False)
                if not confirms:
                    _logger.debug(f"[{self.service_name}] Bybit kline not confirmed yet, skipping: {kline_data.get('symbol', '')}")
                    return
                
                # 解析 K线
                kline = self.ws_client.parse_kline(message)
                if not kline:
                    return
                
                # 从 topic 提取 symbol
                topic = message.topic
                parts = topic.split(".")
                if len(parts) >= 3:
                    symbol = parts[2]
                else:
                    _logger.warning(f"[{self.service_name}] Invalid kline topic: {topic}")
                    return
                
            elif self.exchange == "binance":
                # Binance K线消息：检查 x 字段（是否已关闭）
                data = message.data if hasattr(message, 'data') and message.data else message.raw.get("data", {})
                if not data or data.get("e") != "kline":
                    return
                
                k = data.get("k", {})
                symbol = k.get("s", "").upper()
                if not symbol:
                    return
                
                # 只处理已关闭的 K线（x=true）
                is_closed = k.get("x", False)
                if not is_closed:
                    _logger.debug(f"[{self.service_name}] Binance kline not closed yet, skipping: {symbol}")
                    return
                
                kline = Kline(
                    open_time_ms=int(k.get("t", 0)),
                    close_time_ms=int(k.get("T", 0)),
                    open=float(k.get("o", 0)),
                    high=float(k.get("h", 0)),
                    low=float(k.get("l", 0)),
                    close=float(k.get("c", 0)),
                    volume=float(k.get("v", 0)),
                )
            
            # 更新同步状态
            with self._lock:
                self.sync_status[symbol]["kline_last_update"] = now_ms()
                self.sync_status[symbol]["kline_ok"] = True
            
            # 调用回调保存到数据库
            if self.kline_callback:
                try:
                    self.kline_callback(symbol, kline)
                    _logger.debug(f"[{self.service_name}] Kline saved: {symbol} open_time_ms={kline.open_time_ms}")
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Error in kline callback: {e}", exc_info=True)
                    # 回调失败时，状态仍然标记为 OK（数据已接收）
            
        except Exception as e:
            _logger.error(f"[{self.service_name}] Error processing kline: {e}", exc_info=True)
    
    async def start(self):
        """启动 K线 WebSocket 服务"""
        if self._running:
            return
        
        self._running = True
        _logger.info(f"[{self.service_name}] Starting Kline WebSocket service for {self.exchange}")
        
        # 获取 WebSocket 客户端
        self.ws_client = self._get_ws_client()
        
        # 注册消息回调
        if self.exchange == "bybit":
            self.ws_client.on_message("kline.*", self._on_kline_message)
        elif self.exchange == "binance":
            self.ws_client.on_message("*@kline_*", self._on_kline_message)
        
        # 启动 WebSocket 客户端
        await self.ws_client.start()
        
        # 等待连接建立
        await asyncio.sleep(2)
        
        # 订阅 K线流
        await self._subscribe_kline_streams()
        
        _logger.info(f"[{self.service_name}] Kline WebSocket service started")
    
    async def stop(self):
        """停止 K线 WebSocket 服务"""
        self._running = False
        if self.ws_client:
            await self.ws_client.stop()
        _logger.info(f"[{self.service_name}] Kline WebSocket service stopped")
    
    def get_sync_status(self, symbol: str) -> Dict[str, Any]:
        """获取同步状态"""
        with self._lock:
            status = self.sync_status.get(symbol, {}).copy()
            # 添加连接状态，方便诊断
            status["ws_connected"] = self.is_connected()
            return status
    
    def is_connected(self) -> bool:
        """检查 WebSocket 是否连接"""
        if not self.ws_client:
            return False
        
        if self.exchange == "bybit":
            return hasattr(self.ws_client, 'public_connected') and self.ws_client.public_connected
        elif self.exchange == "binance":
            return hasattr(self.ws_client, 'connected') and self.ws_client.connected
        
        return False
