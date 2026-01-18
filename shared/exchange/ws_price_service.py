"""价格 WebSocket 服务（独立连接）。

专门用于订阅和缓存价格数据，仅用于 Web UI 显示。
价格数据不保存到数据库，只缓存在内存中。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from shared.domain.time import now_ms

_logger = logging.getLogger(__name__)


@dataclass
class LatestPrice:
    """最新价格数据"""
    symbol: str
    price: float
    timestamp_ms: int


class PriceWebSocketService:
    """价格 WebSocket 服务（仅订阅 Ticker 流）。
    
    负责：
    - 订阅所有交易对的 Ticker 流
    - 缓存最新价格（内存）
    - 可选：写入 Redis（供 Web UI 使用）
    - 不保存到数据库
    """
    
    def __init__(
        self,
        exchange: str,
        symbols: List[str],
        *,
        service_name: str = "price-ws",
        redis_client=None,
    ):
        self.exchange = exchange.lower()
        self.symbols = symbols
        self.service_name = service_name
        self.redis_client = redis_client
        
        # 价格缓存（仅内存）
        self.latest_prices: Dict[str, LatestPrice] = {}
        self.sync_status: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # WebSocket 客户端
        self.ws_client = None
        self._running = False
        self._lock = threading.Lock()
        
        # 订阅管理
        self.subscribed_topics: Set[str] = set()
        
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
    
    async def _subscribe_ticker_streams(self):
        """订阅所有交易对的 Ticker 流"""
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
            for symbol in self.symbols:
                # Ticker 流: tickers.{symbol}
                ticker_topic = f"tickers.{symbol}"
                topics.append(ticker_topic)
        
        # Binance 订阅格式（使用组合流）
        elif self.exchange == "binance":
            streams = []
            for symbol in self.symbols:
                symbol_lower = symbol.lower()
                # Ticker 流: {symbol}@ticker
                ticker_stream = f"{symbol_lower}@ticker"
                streams.append(ticker_stream)
            
            # Binance 使用组合流
            if len(streams) <= 1024:
                try:
                    success = await self.ws_client.subscribe_combined(streams)
                    if success:
                        self.subscribed_topics.update(streams)
                        _logger.info(f"[{self.service_name}] Subscribed {len(streams)} ticker streams via combined WebSocket")
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
                    _logger.info(f"[{self.service_name}] Subscribed ticker: {topic}")
                except Exception as e:
                    _logger.error(f"[{self.service_name}] Failed to subscribe {topic}: {e}")
            
            if i + batch_size < len(topics):
                await asyncio.sleep(0.2)
        
        _logger.info(f"[{self.service_name}] Subscribed {len(topics)} ticker topics for {len(self.symbols)} symbols")
    
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
                # Binance 组合流格式: {"stream":"btcusdt@ticker","data":{...}}
                # 单流格式: 直接是数据对象
                if hasattr(message, 'raw') and message.raw:
                    # 组合流格式：从 raw 中提取 stream 和 data
                    raw_data = message.raw
                    if "stream" in raw_data and "data" in raw_data:
                        stream = raw_data.get("stream", "")
                        data = raw_data.get("data", {})
                        # 检查是否是 ticker 流
                        if "@ticker" not in stream:
                            return
                    else:
                        # 单流格式：直接是数据
                        data = raw_data
                else:
                    # 尝试从 message.data 获取
                    data = message.data if hasattr(message, 'data') and message.data else {}
                
                if not data:
                    return
                
                symbol = data.get("s", "").upper()
                price = float(data.get("c", 0))  # 最新成交价（24hrTicker 事件）
            
            if not symbol or price <= 0:
                return
            
            # 更新价格缓存（仅内存，不保存到数据库）
            with self._lock:
                self.latest_prices[symbol] = LatestPrice(
                    symbol=symbol,
                    price=price,
                    timestamp_ms=now_ms(),
                )
                
                # 更新同步状态
                self.sync_status[symbol]["price_last_update"] = now_ms()
                self.sync_status[symbol]["price_ok"] = True
            
            # 可选：写入 Redis 缓存（供 Web UI 使用）
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
        """启动价格 WebSocket 服务"""
        if self._running:
            return
        
        self._running = True
        _logger.info(f"[{self.service_name}] Starting Price WebSocket service for {self.exchange}")
        
        # 获取 WebSocket 客户端
        self.ws_client = self._get_ws_client()
        
        # 注册消息回调
        if self.exchange == "bybit":
            self.ws_client.on_message("tickers.*", self._on_ticker_message)
        elif self.exchange == "binance":
            self.ws_client.on_message("*@ticker", self._on_ticker_message)
        
        # 启动 WebSocket 客户端
        await self.ws_client.start()
        
        # 等待连接建立
        await asyncio.sleep(2)
        
        # 订阅 Ticker 流
        await self._subscribe_ticker_streams()
        
        _logger.info(f"[{self.service_name}] Price WebSocket service started")
    
    async def stop(self):
        """停止价格 WebSocket 服务"""
        self._running = False
        if self.ws_client:
            await self.ws_client.stop()
        _logger.info(f"[{self.service_name}] Price WebSocket service stopped")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        with self._lock:
            price_data = self.latest_prices.get(symbol)
            return price_data.price if price_data else None
    
    def get_all_prices(self) -> Dict[str, float]:
        """获取所有交易对的最新价格"""
        with self._lock:
            return {symbol: data.price for symbol, data in self.latest_prices.items()}
    
    def get_sync_status(self, symbol: str) -> Dict[str, Any]:
        """获取同步状态"""
        with self._lock:
            return self.sync_status.get(symbol, {}).copy()
    
    def is_connected(self) -> bool:
        """检查 WebSocket 是否连接"""
        if not self.ws_client:
            return False
        
        if self.exchange == "bybit":
            return hasattr(self.ws_client, 'public_connected') and self.ws_client.public_connected
        elif self.exchange == "binance":
            return hasattr(self.ws_client, 'connected') and self.ws_client.connected
        
        return False
