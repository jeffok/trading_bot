"""WebSocket 数据同步模块。

使用两个独立的 WebSocket 连接：
1. K线 WebSocket：订阅 K线流，保存到数据库
2. 价格 WebSocket：订阅 Ticker 流，仅缓存到内存（供 Web UI 使用）
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Optional

from shared.exchange.types import Kline
from shared.exchange.ws_kline_service import KlineWebSocketService
from shared.exchange.ws_price_service import PriceWebSocketService

_logger = logging.getLogger(__name__)


class WebSocketSyncManager:
    """WebSocket 同步管理器。
    
    管理两个独立的 WebSocket 服务：
    - K线服务：保存到数据库
    - 价格服务：仅缓存到内存（供 Web UI 使用）
    """
    
    def __init__(
        self,
        exchange: str,
        symbols: list[str],
        interval_minutes: int,
        kline_callback: Callable[[str, Kline], None],
        *,
        redis_client=None,
        service_name: str = "data-syncer-ws",
    ):
        self.exchange = exchange
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.kline_callback = kline_callback
        self.redis_client = redis_client
        self.service_name = service_name
        
        self.kline_service: Optional[KlineWebSocketService] = None
        self.price_service: Optional[PriceWebSocketService] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def _run_async_loop(self):
        """在独立线程中运行异步事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            # 创建 K线 WebSocket 服务（保存到数据库）
            self.kline_service = KlineWebSocketService(
                exchange=self.exchange,
                symbols=self.symbols,
                interval_minutes=self.interval_minutes,
                kline_callback=self.kline_callback,
                service_name=f"{self.service_name}-kline",
            )
            
            # 创建价格 WebSocket 服务（仅缓存到内存）
            self.price_service = PriceWebSocketService(
                exchange=self.exchange,
                symbols=self.symbols,
                service_name=f"{self.service_name}-price",
                redis_client=self.redis_client,
            )
            
            # 启动两个服务
            async def start_services():
                await asyncio.gather(
                    self.kline_service.start(),
                    self.price_service.start(),
                )
            
            self._loop.run_until_complete(start_services())
            
            # 保持运行
            self._loop.run_forever()
        except Exception as e:
            _logger.error(f"[{self.service_name}] WebSocket loop error: {e}", exc_info=True)
        finally:
            async def stop_services():
                if self.kline_service:
                    try:
                        await self.kline_service.stop()
                    except Exception:
                        pass
                if self.price_service:
                    try:
                        await self.price_service.stop()
                    except Exception:
                        pass
            
            if self._loop and not self._loop.is_closed():
                self._loop.run_until_complete(stop_services())
            self._loop.close()
    
    def start(self):
        """启动 WebSocket 同步服务（在独立线程中）"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._run_async_loop,
            name="ws-sync",
            daemon=True,
        )
        self._thread.start()
        _logger.info(f"[{self.service_name}] WebSocket sync manager started (Kline + Price)")
    
    def stop(self):
        """停止 WebSocket 同步服务"""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        _logger.info(f"[{self.service_name}] WebSocket sync manager stopped")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格（从价格服务）"""
        if self.price_service:
            return self.price_service.get_latest_price(symbol)
        return None
    
    def get_all_prices(self) -> dict[str, float]:
        """获取所有交易对的最新价格（从价格服务）"""
        if self.price_service:
            return self.price_service.get_all_prices()
        return {}
    
    def get_sync_status(self, symbol: str) -> dict:
        """获取同步状态"""
        status = {}
        if self.kline_service:
            # K线服务状态
            kline_status = self.kline_service.get_sync_status(symbol)
            status.update(kline_status)
            status["kline_connected"] = self.kline_service.is_connected()
        if self.price_service:
            # 价格服务状态
            price_status = self.price_service.get_sync_status(symbol)
            status.update(price_status)
            status["price_connected"] = self.price_service.is_connected()
        return status
    
    def is_connected(self) -> bool:
        """检查 WebSocket 是否连接（K线服务）"""
        return self.kline_service is not None and self.kline_service.is_connected()
