"""Redis leader election helper.

Design:
- Use SET key value NX EX ttl to acquire leadership.
- Renew leadership by checking value matches instance_id and then EXPIRE.
- Followers do not perform scheduled work; they only emit heartbeats and metrics.
"""

from __future__ import annotations

import time
from typing import Optional

import redis

_RENEW_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("expire", KEYS[1], ARGV[2])
else
  return 0
end
"""

_RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
"""

class LeaderElector:
    def __init__(self, r: redis.Redis, *, key: str, instance_id: str, ttl_seconds: int = 30, renew_interval_seconds: int = 10):
        self.r = r
        self.key = key
        self.instance_id = instance_id
        self.ttl_seconds = int(ttl_seconds)
        self.renew_interval_seconds = int(renew_interval_seconds)
        self._last_renew_ts: float = 0.0
        self._is_leader: bool = False

    def get_leader(self) -> Optional[str]:
        try:
            v = self.r.get(self.key)
            return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else (str(v) if v else None)
        except Exception:
            return None

    def try_acquire(self) -> bool:
        try:
            ok = self.r.set(self.key, self.instance_id, nx=True, ex=self.ttl_seconds)
            if ok:
                self._is_leader = True
                self._last_renew_ts = time.time()
                return True
            return False
        except Exception:
            return False

    def renew(self) -> bool:
        try:
            res = self.r.eval(_RENEW_LUA, 1, self.key, self.instance_id, str(self.ttl_seconds))
            ok = int(res) == 1
            if ok:
                self._is_leader = True
                self._last_renew_ts = time.time()
            else:
                self._is_leader = False
            return ok
        except Exception:
            self._is_leader = False
            return False

    def release(self) -> bool:
        try:
            res = self.r.eval(_RELEASE_LUA, 1, self.key, self.instance_id)
            self._is_leader = False
            return int(res) == 1
        except Exception:
            self._is_leader = False
            return False

    def is_leader(self) -> bool:
        return bool(self._is_leader)

    def ensure(self) -> bool:
        """Ensure a leader exists and renew if we are leader.

        Returns True iff current instance should act as leader.
        """
        now = time.time()
        if self._is_leader:
            # renew periodically
            if now - self._last_renew_ts >= self.renew_interval_seconds:
                return self.renew()
            return True

        # not leader: attempt acquire; if fail, check if current leader key points to us (rare)
        if self.try_acquire():
            return True
        cur = self.get_leader()
        self._is_leader = (cur == self.instance_id)
        return self._is_leader
