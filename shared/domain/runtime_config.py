from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from shared.config import Settings
from shared.db import PostgreSQL


def _parse_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_int(v: Optional[str], default: int) -> int:
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default




def _parse_float(v: Optional[str], default: float) -> float:
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default
def _parse_symbols(value: Optional[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    raw = str(value).strip()
    if not raw:
        return tuple()
    parts = [p.strip().upper() for p in re.split(r"[\s,]+", raw) if p.strip()]
    # keep order, de-dup
    out = []
    seen = set()
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return tuple(out)


@dataclass
class RuntimeConfig:
    """Runtime config backed by DB table `system_config`.

    - If SYMBOLS is not set in DB (or empty), fallback to env settings.symbols.
    - This is intentionally small; it can be extended as the spec requires.
    """

    symbols: Tuple[str, ...]
    symbols_from_db: bool
    halt_trading: bool
    emergency_exit: bool
    use_protective_stop_order: bool
    stop_order_poll_seconds: int
    stop_arm_max_retries: int
    stop_arm_backoff_base_seconds: float
    stop_rearm_max_attempts: int
    stop_rearm_cooldown_seconds: int
    last_refresh_ms: int = 0

    @staticmethod
    def _fetch_keys(db: PostgreSQL, keys: Tuple[str, ...]) -> Dict[str, Optional[str]]:
        if not keys:
            return {}
        placeholders = ",".join(["%s"] * len(keys))
        rows = db.fetch_all(f'SELECT "key", "value" FROM system_config WHERE "key" IN ({placeholders})', keys)
        m: Dict[str, Optional[str]] = {k: None for k in keys}
        for r in rows or []:
            k = str(r.get("key") or "")
            if k:
                m[k] = None if r.get("value") is None else str(r.get("value"))
        return m

    @classmethod
    def load(cls, db: PostgreSQL, settings: Settings) -> "RuntimeConfig":
        m = cls._fetch_keys(
            db,
            (
            "SYMBOLS",
            "HALT_TRADING",
            "EMERGENCY_EXIT",
            "USE_PROTECTIVE_STOP_ORDER",
            "STOP_ORDER_POLL_SECONDS",
            "STOP_ARM_MAX_RETRIES",
            "STOP_ARM_BACKOFF_BASE_SECONDS",
            "STOP_REARM_MAX_ATTEMPTS",
            "STOP_REARM_COOLDOWN_SECONDS",
        ),
        )
        db_symbols = _parse_symbols(m.get("SYMBOLS"))
        symbols = db_symbols if db_symbols else tuple(settings.symbols)
        symbols_from_db = bool(db_symbols)

        return cls(
            symbols=symbols,
            symbols_from_db=symbols_from_db,
            halt_trading=_parse_bool(m.get("HALT_TRADING"), default=False),
            emergency_exit=_parse_bool(m.get("EMERGENCY_EXIT"), default=False),
            use_protective_stop_order=_parse_bool(m.get("USE_PROTECTIVE_STOP_ORDER"), default=settings.use_protective_stop_order),
            stop_order_poll_seconds=_parse_int(m.get("STOP_ORDER_POLL_SECONDS"), default=int(settings.stop_order_poll_seconds)),
            stop_arm_max_retries=_parse_int(m.get("STOP_ARM_MAX_RETRIES"), default=int(settings.stop_arm_max_retries)),
            stop_arm_backoff_base_seconds=_parse_float(m.get("STOP_ARM_BACKOFF_BASE_SECONDS"), default=float(settings.stop_arm_backoff_base_seconds)),
            stop_rearm_max_attempts=_parse_int(m.get("STOP_REARM_MAX_ATTEMPTS"), default=int(settings.stop_rearm_max_attempts)),
            stop_rearm_cooldown_seconds=_parse_int(m.get("STOP_REARM_COOLDOWN_SECONDS"), default=int(settings.stop_rearm_cooldown_seconds)),
            last_refresh_ms=int(time.time() * 1000),
        )

    def refresh(self, db: PostgreSQL, settings: Settings) -> Dict[str, Any]:
        before = (
            self.symbols,
            self.symbols_from_db,
            self.halt_trading,
            self.emergency_exit,
            self.use_protective_stop_order,
            self.stop_order_poll_seconds,
            self.stop_arm_max_retries,
            self.stop_arm_backoff_base_seconds,
            self.stop_rearm_max_attempts,
            self.stop_rearm_cooldown_seconds,
        )

        m = self._fetch_keys(
            db,
            (
            "SYMBOLS",
            "HALT_TRADING",
            "EMERGENCY_EXIT",
            "USE_PROTECTIVE_STOP_ORDER",
            "STOP_ORDER_POLL_SECONDS",
            "STOP_ARM_MAX_RETRIES",
            "STOP_ARM_BACKOFF_BASE_SECONDS",
            "STOP_REARM_MAX_ATTEMPTS",
            "STOP_REARM_COOLDOWN_SECONDS",
        ),
        )
        db_symbols = _parse_symbols(m.get("SYMBOLS"))
        self.symbols = db_symbols if db_symbols else tuple(settings.symbols)
        self.symbols_from_db = bool(db_symbols)
        self.halt_trading = _parse_bool(m.get("HALT_TRADING"), default=False)
        self.emergency_exit = _parse_bool(m.get("EMERGENCY_EXIT"), default=False)
        self.use_protective_stop_order = _parse_bool(
            m.get("USE_PROTECTIVE_STOP_ORDER"), default=settings.use_protective_stop_order
        )
        self.stop_order_poll_seconds = _parse_int(
            m.get("STOP_ORDER_POLL_SECONDS"), default=int(settings.stop_order_poll_seconds)
        )
        self.stop_arm_max_retries = _parse_int(m.get("STOP_ARM_MAX_RETRIES"), default=int(settings.stop_arm_max_retries))
        self.stop_arm_backoff_base_seconds = _parse_float(
            m.get("STOP_ARM_BACKOFF_BASE_SECONDS"), default=float(settings.stop_arm_backoff_base_seconds)
        )
        self.stop_rearm_max_attempts = _parse_int(m.get("STOP_REARM_MAX_ATTEMPTS"), default=int(settings.stop_rearm_max_attempts))
        self.stop_rearm_cooldown_seconds = _parse_int(
            m.get("STOP_REARM_COOLDOWN_SECONDS"), default=int(settings.stop_rearm_cooldown_seconds)
        )
        self.last_refresh_ms = int(time.time() * 1000)

        after = (
            self.symbols,
            self.symbols_from_db,
            self.halt_trading,
            self.emergency_exit,
            self.use_protective_stop_order,
            self.stop_order_poll_seconds,
            self.stop_arm_max_retries,
            self.stop_arm_backoff_base_seconds,
            self.stop_rearm_max_attempts,
            self.stop_rearm_cooldown_seconds,
        )

        changes: Dict[str, Any] = {}
        if before[0] != after[0] or before[1] != after[1]:
            changes["symbols"] = {"from_db": self.symbols_from_db, "symbols": list(self.symbols)}
        if before[2] != after[2]:
            changes["halt_trading"] = self.halt_trading
        if before[3] != after[3]:
            changes["emergency_exit"] = self.emergency_exit
        if before[4] != after[4]:
            changes["use_protective_stop_order"] = self.use_protective_stop_order
        if before[5] != after[5]:
            changes["stop_order_poll_seconds"] = self.stop_order_poll_seconds
        if before[6] != after[6]:
            changes["stop_arm_max_retries"] = self.stop_arm_max_retries
        if before[7] != after[7]:
            changes["stop_arm_backoff_base_seconds"] = self.stop_arm_backoff_base_seconds
        if before[8] != after[8]:
            changes["stop_rearm_max_attempts"] = self.stop_rearm_max_attempts
        if before[9] != after[9]:
            changes["stop_rearm_cooldown_seconds"] = self.stop_rearm_cooldown_seconds
        return changes
