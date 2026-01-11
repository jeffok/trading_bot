from __future__ import annotations

from .rate_limiter import AdaptiveRateLimiter
from .binance import BinanceUsdtFuturesClient
from .bybit import BybitV5LinearClient
from .paper import PaperExchange


def make_exchange(settings, *, metrics=None, service_name: str = "unknown"):
    """创建交易所客户端（本项目仅支持逐仓合约：Binance USDT-M Futures / Bybit Linear）。

    运行期：二选一（EXCHANGE=binance 或 bybit 或 paper）
    """
    limiter = AdaptiveRateLimiter(metrics=metrics, exchange=settings.exchange.lower())
    ex = settings.exchange.lower()

    if ex == "binance":
        return BinanceUsdtFuturesClient(
            base_url=settings.binance_base_url,
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            recv_window=settings.binance_recv_window,
            leverage=settings.futures_leverage,
            limiter=limiter,
            metrics=metrics,
            service_name=service_name,
        )

    if ex == "bybit":
        return BybitV5LinearClient(
            base_url=settings.bybit_base_url,
            api_key=settings.bybit_api_key,
            api_secret=settings.bybit_api_secret,
            recv_window=settings.bybit_recv_window,
            leverage=settings.futures_leverage,
            position_idx=settings.bybit_position_idx,
            limiter=limiter,
            metrics=metrics,
            service_name=service_name,
        )

    return PaperExchange(starting_usdt=settings.paper_starting_usdt, fee_pct=settings.paper_fee_pct)
