
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class Budget:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_refill: float
    backoff_until: float = 0.0

class AdaptiveRateLimiter:
    def __init__(self):
        self.budgets: Dict[str, Budget] = {}

    def ensure_budget(self, key: str, capacity: float, refill_per_sec: float) -> None:
        if key not in self.budgets:
            now = time.time()
            self.budgets[key] = Budget(capacity, refill_per_sec, capacity, now)

    def acquire(self, key: str, cost: float = 1.0) -> None:
        b = self.budgets[key]
        now = time.time()

        if now < b.backoff_until:
            time.sleep(b.backoff_until - now)

        elapsed = now - b.last_refill
        if elapsed > 0:
            b.tokens = min(b.capacity, b.tokens + elapsed * b.refill_per_sec)
            b.last_refill = now

        if b.tokens < cost:
            need = cost - b.tokens
            sleep_s = need / b.refill_per_sec if b.refill_per_sec > 0 else 0.2
            time.sleep(max(0.01, sleep_s))
            return self.acquire(key, cost)

        b.tokens -= cost

    def backoff(self, key: str, seconds: float) -> None:
        b = self.budgets[key]
        b.backoff_until = max(b.backoff_until, time.time() + seconds)
