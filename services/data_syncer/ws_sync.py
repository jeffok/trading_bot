"""WebSocket 数据同步模块。

使用公共 WebSocket 接口实时接收 K线数据，替代 REST API 轮询。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Optional

from shared.exchange.types import Kline
from shared.exchange.ws_market_data import MarketDataWebSocketService

_logger = logging.getLogger(__name__)


class WebSocketSyncManager:
    """WebSocket 同步管理器。
    
    在独立线程中运行异步 WebSocket 服务，与主循环解耦。
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
        
        self.ws_service: Optional[MarketDataWebSocketService] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def _run_async_loop(self):
        """在独立线程中运行异步事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            # 创建 WebSocket 服务
            self.ws_service = MarketDataWebSocketService(
                exchange=self.exchange,
                symbols=self.symbols,
                interval_minutes=self.interval_minutes,
                service_name=self.service_name,
                redis_client=self.redis_client,
            )
            
            # 设置 K线回调
            self.ws_service.set_kline_callback(self.kline_callback)
            
            # 启动服务
            self._loop.run_until_complete(self.ws_service.start())
            
            # 保持运行
            self._loop.run_forever()
        except Exception as e:
            _logger.error(f"[{self.service_name}] WebSocket loop error: {e}", exc_info=True)
        finally:
            if self.ws_service:
                try:
                    self._loop.run_until_complete(self.ws_service.stop())
                except Exception:
                    pass
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
        _logger.info(f"[{self.service_name}] WebSocket sync manager started")
    
    def stop(self):
        """停止 WebSocket 同步服务"""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        _logger.info(f"[{self.service_name}] WebSocket sync manager stopped")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        if self.ws_service:
            return self.ws_service.get_latest_price(symbol)
        return None
    
    def get_latest_kline(self, symbol: str) -> Optional[Kline]:
        """获取最新K线"""
        if self.ws_service:
            return self.ws_service.get_latest_kline(symbol)
        return None
    
    def get_sync_status(self, symbol: str) -> dict:
        """获取同步状态"""
        if self.ws_service:
            return self.ws_service.get_sync_status(symbol)
        return {}
    
    def get_all_prices(self) -> dict[str, float]:
        """获取所有交易对的最新价格"""
        if self.ws_service:
            return self.ws_service.get_all_prices()
        return {}
    
    def is_connected(self) -> bool:
        """检查 WebSocket 是否连接"""
        return self.ws_service is not None and self.ws_service.ws_client is not None and (
            (hasattr(self.ws_service.ws_client, 'public_connected') and self.ws_service.ws_client.public_connected) or
            (hasattr(self.ws_service.ws_client, 'connected') and self.ws_service.ws_client.connected)
        )
