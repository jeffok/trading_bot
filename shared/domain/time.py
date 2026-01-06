
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo

HK = ZoneInfo("Asia/Hong_Kong")

def now_ms() -> int:
    return int(shared.domain.time() * 1000)

def next_tick_sleep_seconds(interval_seconds: int) -> float:
    now = datetime.now(HK)
    epoch = now.timestamp()
    next_epoch = ((int(epoch) // interval_seconds) + 1) * interval_seconds
    return max(0.0, next_epoch - epoch)
