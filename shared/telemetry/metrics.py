
from __future__ import annotations
from prometheus_client import Counter, Gauge, Histogram

class Metrics:
    def __init__(self, service: str):
        self.service = service
        self.orders_total = Counter("orders_total", "Total orders", ("service","exchange","symbol","status"))
        self.exchange_requests_total = Counter("exchange_requests_total", "Exchange HTTP requests", ("service","exchange","endpoint","status"))
        self.exchange_latency_seconds = Histogram("exchange_latency_seconds", "Exchange latency", ("service","exchange","endpoint"))
        self.data_sync_lag_ms = Gauge("data_sync_lag_ms", "Data sync lag ms", ("service","symbol","interval_minutes"))
        self.last_tick_success = Gauge("last_tick_success", "Last tick success", ("service","symbol"))
