
"""Redis distributed lock (SET NX PX + safe unlock)."""

from __future__ import annotations
import contextlib, time, uuid
from typing import Iterator
import redis

_UNLOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
"""

@contextlib.contextmanager
def distributed_lock(r: redis.Redis, key: str, ttl_ms: int = 55_000, wait_ms: int = 3_000, poll_ms: int = 100) -> Iterator[bool]:
    token = str(uuid.uuid4())
    deadline = time.time() + (wait_ms / 1000.0)
    acquired = False
    while time.time() < deadline:
        if r.set(key, token, nx=True, px=ttl_ms):
            acquired = True
            break
        time.sleep(poll_ms / 1000.0)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                r.eval(_UNLOCK_LUA, 1, key, token)
            except Exception:
                pass
