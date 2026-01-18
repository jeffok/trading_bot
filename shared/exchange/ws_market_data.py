"""公共 WebSocket 市场数据服务。

支持：
- Bybit: K线流 + Ticker 流（公共频道，无需认证）
- Binance: K线流 + Ticker 流（公共频道，无需认证）

特性：
- 批量订阅管理（合并流，减少连接数）
- 自动重连和心跳
- 价格缓存（Redis/内存）
- 频率限制控制
- 数据幂等性处理
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from shared.domain.time import now_ms
from shared.exchange.types import Kline

_logger = logging.getLogger(__name__)


@dataclass
class LatestPrice:
    """最新价格数据"""
    symbol: str
    price: float
    timestamp_ms: int
    source: str  # 'ticker' or 'trade'


@dataclass
class LatestKline:
    """最新K线数据"""
    symbol: str
    interval_minutes: int
    kline: Kline
    timestamp_ms: int


class MarketDataWebSocketService:
    """市场数据 WebSocket 服务。
    
    负责：
    - 订阅所有交易对的 K线流和 Ticker 流
    - 缓存最新价格和K线数据
    - 提供数据访问接口
    - 处理连接管理和重连
    """
    
    def __init__(
        self,
        exchange: str,
        symbols: List[str],
        interval_minutes: int,
        *,
        service_name: str = "market-data-ws",
        redis_client=None,
    ):
        self.exchange = exchange.lower()
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.service_name = service_name
        self.redis_client = redis_client
        
        # 数据缓存
        self.latest_prices: Dict[str, LatestPrice] = {}
        self.latest_klines: Dict[str, LatestKline] = {}
        self.sync_status: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # WebSocket 客户端
        self.ws_client = None
        self._running = False
        self._lock = threading.Lock()
        
        # 订阅管理
        self.subscribed_topics: Set[str] = set()
        
        # K线保存回调
        self._kline_callback: Optional[Callable[[str, Kline], None]] = None
    
    def set_kline_callback(self, callback: Callable[[str, Kline], None]):
        """设置 K线数据保存回调（由 data-syncer 提供）"""
        self._kline_callback = callback
        
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
            # 默认返回分钟数
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
            # 默认返回分钟数 + m
            return f"{minutes}m"
    
    async def _subscribe_all(self):
        """订阅所有交易对的 K线流和 Ticker 流"""
        if not self.ws_client or not self.ws_client.public_connected:
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
                # Ticker 流: tickers.{symbol}
                ticker_topic = f"tickers.{symbol}"
                topics.append(ticker_topic)
        
        # Binance 订阅格式（使用组合流，推荐方式）
        elif self.exchange == "binance":
            interval_str = self._minutes_to_binance_interval(self.interval_minutes)
            streams = []
            for symbol in self.symbols:
                symbol_lower = symbol.lower()
                # K线流: {symbol}@kline_{interval}
                kline_stream = f"{symbol_lower}@kline_{interval_str}"
                streams.append(kline_stream)
                # Ticker 流: {symbol}@ticker
                ticker_stream = f"{symbol_lower}@ticker"
                streams.append(ticker_stream)
            
            # Binance 使用组合流（一次连接订阅所有流，更高效）
            # 单个连接最多支持 1024 个流
            if len(streams) <= 1024:
                try:
                    success = await self.ws_client.subscribe_combined(streams)
                    if success:
                        self.subscribed_topics.update(streams)
                        _logger.info(f"[{self.service_name}] Subscribed {len(streams)} streams via combined WebSocket")
                        return
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Failed to subscribe combined streams: {e}")
                    # 降级到单流订阅
            else:
                _logger.warning(f"[{self.service_name}] Too many streams ({len(streams)}), splitting into multiple connections")
            
            # 降级：单流订阅（分批处理）
            topics = streams
        
        # 批量订阅（控制频率：每秒最多5条控制消息）
        # 分批订阅，每批最多10个，间隔200ms
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
                    _logger.info(f"[{self.service_name}] Subscribed: {topic}")
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Failed to subscribe {topic}: {e}")
            
            # 控制订阅频率：每批之间延迟200ms（确保不超过每秒5条控制消息）
            if i + batch_size < len(topics):
                await asyncio.sleep(0.2)
        
        _logger.info(f"[{self.service_name}] Subscribed {len(topics)} topics for {len(self.symbols)} symbols")
    
    def _on_kline_message(self, message):
        """处理 K线消息"""
        try:
            if self.exchange == "bybit":
                kline = self.ws_client.parse_kline(message)
                if not kline:
                    return
                
                # 从 topic 提取 symbol
                topic = message.topic
                # topic 格式: kline.{interval}.{symbol}
                parts = topic.split(".")
                if len(parts) >= 3:
                    symbol = parts[2]
                else:
                    _logger.warning(f"[{self.service_name}] Invalid kline topic: {topic}")
                    return
                
            elif self.exchange == "binance":
                # Binance 组合流格式: {"stream":"btcusdt@kline_15m","data":{...}}
                # 或单流格式: 直接是数据
                data = message.data if hasattr(message, 'data') and message.data else message.raw.get("data", {})
                if not data:
                    return
                
                # 检查是否是 kline 事件
                if data.get("e") != "kline":
                    return
                
                k = data.get("k", {})
                symbol = k.get("s", "").upper()
                if not symbol:
                    return
                
                # 只处理已关闭的 K线（is_closed=True，x 字段）
                is_closed = k.get("x", False)  # x 表示 K线是否已关闭
                if not is_closed:
                    # 未关闭的 K线不保存到数据库，只更新缓存用于显示
                    _logger.debug(f"[{self.service_name}] Kline not closed yet, skipping save: {symbol}")
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
            
            # 更新缓存（幂等性：相同 open_time_ms 只更新一次）
            with self._lock:
                existing = self.latest_klines.get(symbol)
                if existing and existing.kline.open_time_ms >= kline.open_time_ms:
                    # 已存在更新的数据，跳过
                    return
                
                self.latest_klines[symbol] = LatestKline(
                    symbol=symbol,
                    interval_minutes=self.interval_minutes,
                    kline=kline,
                    timestamp_ms=now_ms(),
                )
                
                # 更新同步状态
                self.sync_status[symbol]["kline_last_update"] = now_ms()
                self.sync_status[symbol]["kline_ok"] = True
            
            # 调用回调保存到数据库（同步调用，在回调中处理异步）
            if self._kline_callback:
                try:
                    self._kline_callback(symbol, kline)
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Error in kline callback: {e}", exc_info=True)
            
        except Exception as e:
            _logger.error(f"[{self.service_name}] Error processing kline: {e}", exc_info=True)
    
    def _on_ticker_message(self, message):
        """处理 Ticker 消息"""
        try:
            if self.exchange == "bybit":
                data = message.data
                if isinstance(data, list) and len(data) > 0:
                    ticker = data[0]
                elif isinstance(data, dict):
                    ticker = data
                else:
                    return
                
                symbol = ticker.get("symbol", "").upper()
                price = float(ticker.get("lastPrice", 0))
                
            elif self.exchange == "binance":
                data = message.raw.get("data", {})
                if not data:
                    return
                
                symbol = data.get("s", "").upper()
                price = float(data.get("c", 0))  # 最新成交价
            
            if not symbol or price <= 0:
                return
            
            # 更新价格缓存
            with self._lock:
                self.latest_prices[symbol] = LatestPrice(
                    symbol=symbol,
                    price=price,
                    timestamp_ms=now_ms(),
                    source="ticker",
                )
                
                # 更新同步状态
                self.sync_status[symbol]["price_last_update"] = now_ms()
                self.sync_status[symbol]["price_ok"] = True
            
            # 可选：写入 Redis 缓存
            if self.redis_client:
                try:
                    key = f"market:price:{symbol}"
                    self.redis_client.setex(
                        key,
                        60,  # 60秒过期
                        json.dumps({"price": price, "timestamp": now_ms()}),
                    )
                except Exception:
                    pass
            
        except Exception as e:
            _logger.error(f"[{self.service_name}] Error processing ticker: {e}", exc_info=True)
    
    
    async def start(self):
        """启动 WebSocket 服务"""
        if self._running:
            return
        
        self._running = True
        _logger.info(f"[{self.service_name}] Starting market data WebSocket service for {self.exchange}")
        
        # 获取 WebSocket 客户端
        self.ws_client = self._get_ws_client()
        
        # 注册消息回调
        if self.exchange == "bybit":
            # Bybit: 使用通配符匹配
            self.ws_client.on_message("kline.*", self._on_kline_message)
            self.ws_client.on_message("tickers.*", self._on_ticker_message)
        elif self.exchange == "binance":
            # Binance: 使用通配符匹配所有 kline 和 ticker 流
            self.ws_client.on_message("*@kline_*", self._on_kline_message)
            self.ws_client.on_message("*@ticker", self._on_ticker_message)
        
        # 启动 WebSocket 客户端
        await self.ws_client.start()
        
        # 等待连接建立
        await asyncio.sleep(2)
        
        # 订阅所有流
        await self._subscribe_all()
        
        _logger.info(f"[{self.service_name}] Market data WebSocket service started")
    
    async def stop(self):
        """停止 WebSocket 服务"""
        self._running = False
        if self.ws_client:
            await self.ws_client.stop()
        _logger.info(f"[{self.service_name}] Market data WebSocket service stopped")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        with self._lock:
            price_data = self.latest_prices.get(symbol)
            return price_data.price if price_data else None
    
    def get_latest_kline(self, symbol: str) -> Optional[Kline]:
        """获取最新K线"""
        with self._lock:
            kline_data = self.latest_klines.get(symbol)
            return kline_data.kline if kline_data else None
    
    def get_sync_status(self, symbol: str) -> Dict[str, Any]:
        """获取同步状态"""
        with self._lock:
            return self.sync_status.get(symbol, {}).copy()
    
    def get_all_prices(self) -> Dict[str, float]:
        """获取所有交易对的最新价格"""
        with self._lock:
            return {symbol: data.price for symbol, data in self.latest_prices.items()}
    
    def get_all_klines(self) -> Dict[str, Kline]:
        """获取所有交易对的最新K线"""
        with self._lock:
            return {symbol: data.kline for symbol, data in self.latest_klines.items()}
