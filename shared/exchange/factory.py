
from __future__ import annotations
from .rate_limiter import AdaptiveRateLimiter
from .binance import BinanceSpotClient
from .bybit import BybitV5SpotClient
from .paper import PaperExchange

def make_exchange(settings, *, metrics=None, service_name: str = "unknown"):
    limiter = AdaptiveRateLimiter()
    ex = settings.exchange.lower()

    if ex == "binance":
        return BinanceSpotClient(
            base_url=settings.binance_base_url,
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            recv_window=settings.binance_recv_window,
            limiter=limiter,
            metrics=metrics,
            service_name=service_name,
        )
    if ex == "bybit":
        return BybitV5SpotClient(
            base_url=settings.bybit_base_url,
            api_key=settings.bybit_api_key,
            api_secret=settings.bybit_api_secret,
            recv_window=settings.bybit_recv_window,
            limiter=limiter,
            metrics=metrics,
            service_name=service_name,
        )

    return PaperExchange(starting_usdt=settings.paper_starting_usdt, fee_pct=settings.paper_fee_pct)
