from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Budget:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_refill: float
    backoff_until: float = 0.0
    consecutive_rate_limits: int = 0


class AdaptiveRateLimiter:
    """Adaptive rate limiter with per-key budgets and backoff.

    Keys are typically logical groups: market_data / account / order.

    - Token bucket per key
    - 429/418: Retry-After if present else exponential backoff with jitter
    - Best-effort soft backoff via headers
    - Optional Prometheus metrics via shared.telemetry.metrics.Metrics
    """

    def __init__(
        self,
        *,
        metrics: Optional[Any] = None,
        exchange: str = "",
        severe_threshold: int = 8,
        max_backoff_seconds: float = 60.0,
        jitter_ratio: float = 0.2,
    ) -> None:
        self.budgets: Dict[str, Budget] = {}
        self.metrics = metrics
        self.exchange = exchange
        self.severe_threshold = severe_threshold
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter_ratio = jitter_ratio

    def ensure_budget(self, key: str, rps: float, burst: float) -> None:
        now = time.time()
        if key in self.budgets:
            b = self.budgets[key]
            b.capacity = float(burst)
            b.refill_per_sec = float(rps)
            b.tokens = min(b.tokens, b.capacity)
            return
        self.budgets[key] = Budget(
            capacity=float(burst),
            refill_per_sec=float(rps),
            tokens=float(burst),
            last_refill=now,
        )

    def _refill(self, b: Budget, now: float) -> None:
        elapsed = max(0.0, now - b.last_refill)
        if elapsed <= 0:
            return
        b.tokens = min(b.capacity, b.tokens + elapsed * b.refill_per_sec)
        b.last_refill = now

    def _observe(self, metric_name: str, labels: tuple, value: float) -> None:
        if self.metrics is None:
            return
        m = getattr(self.metrics, metric_name, None)
        if m is None:
            return
        try:
            m.labels(*labels).observe(value)
        except Exception:
            pass

    def _inc(self, metric_name: str, labels: tuple) -> None:
        if self.metrics is None:
            return
        m = getattr(self.metrics, metric_name, None)
        if m is None:
            return
        try:
            m.labels(*labels).inc()
        except Exception:
            pass

    def acquire(self, key: str, cost: float = 1.0) -> None:
        if key not in self.budgets:
            self.ensure_budget(key, rps=2.0, burst=2.0)

        b = self.budgets[key]
        now = time.time()

        # backoff gate
        if b.backoff_until > now:
            sleep_s = b.backoff_until - now
            self._observe("rate_limit_backoff_seconds", (self.metrics.service if self.metrics else "unknown", self.exchange, key), sleep_s)
            time.sleep(sleep_s)
            now = time.time()

        self._refill(b, now)

        if b.tokens < cost:
            need = cost - b.tokens
            wait_s = need / b.refill_per_sec if b.refill_per_sec > 0 else 0.25
            wait_s = max(0.01, min(wait_s, 2.0))
            self._observe("rate_limit_wait_seconds", (self.metrics.service if self.metrics else "unknown", self.exchange, key), wait_s)
            time.sleep(wait_s)
            return self.acquire(key, cost)

        b.tokens -= cost

    def feedback_ok(self, key: str, headers: Optional[Dict[str, Any]] = None) -> None:
        if key not in self.budgets:
            return
        b = self.budgets[key]
        b.consecutive_rate_limits = 0

        if not headers:
            return

        # Honor Retry-After even on 200s (best-effort)
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra:
            try:
                seconds = float(ra)
                if seconds > 0:
                    self._apply_backoff(key, seconds)
            except Exception:
                pass

    def _apply_backoff(self, key: str, seconds: float) -> None:
        b = self.budgets[key]
        b.backoff_until = max(b.backoff_until, time.time() + float(seconds))

    def feedback_rate_limited(
        self,
        key: str,
        *,
        retry_after_seconds: Optional[float] = None,
        status_code: int = 429,
    ) -> Dict[str, Any]:
        if key not in self.budgets:
            self.ensure_budget(key, rps=2.0, burst=2.0)
        b = self.budgets[key]
        b.consecutive_rate_limits += 1

        self._inc("rate_limit_429_total", (self.metrics.service if self.metrics else "unknown", self.exchange, key, str(status_code)))

        # compute backoff
        if retry_after_seconds is not None and retry_after_seconds > 0:
            backoff = float(retry_after_seconds)
        else:
            base = 0.5
            backoff = min(self.max_backoff_seconds, base * (2 ** max(0, b.consecutive_rate_limits - 1)))
            jitter = backoff * self.jitter_ratio
            backoff = max(0.1, backoff + random.uniform(-jitter, jitter))

        self._apply_backoff(key, backoff)
        self._inc("rate_limit_retries_total", (self.metrics.service if self.metrics else "unknown", self.exchange, key))

        severe = b.consecutive_rate_limits >= self.severe_threshold
        return {"backoff_seconds": backoff, "consecutive": b.consecutive_rate_limits, "severe": severe}
