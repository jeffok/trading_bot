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

    - If SYMBOLS is not set in DB (or empty), fallback to env settings.symbols (if provided).
    - All config values (except SYMBOLS) are read from database only, with hardcoded defaults if not set.
    - Config values are hot-reloadable (refreshed every RUNTIME_CONFIG_REFRESH_SECONDS).
    - Use Web UI (/admin/ui) or API (/admin/update_config) to update configuration.
    """

    # 交易对
    symbols: Tuple[str, ...]
    symbols_from_db: bool
    
    # 控制开关
    halt_trading: bool
    emergency_exit: bool
    
    # 停止订单配置
    use_protective_stop_order: bool
    stop_order_poll_seconds: int
    stop_arm_max_retries: int
    stop_arm_backoff_base_seconds: float
    stop_rearm_max_attempts: int
    stop_rearm_cooldown_seconds: int
    
    # 策略参数（Setup B）
    setup_b_adx_min: float
    setup_b_vol_ratio_min: float
    setup_b_ai_score_min: float
    hard_stop_loss_pct: float
    
    # 风控参数
    account_equity_usdt: float
    risk_budget_pct: float
    max_drawdown_pct: float
    max_concurrent_positions: int
    min_order_usdt: float
    
    # AI参数
    ai_enabled: bool
    ai_weight: float
    ai_lr: float
    ai_min_samples: int
    
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
    def load(cls, db: PostgreSQL, settings: Settings = None) -> "RuntimeConfig":
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
            # 策略参数
            "SETUP_B_ADX_MIN",
            "SETUP_B_VOL_RATIO_MIN",
            "SETUP_B_AI_SCORE_MIN",
            "HARD_STOP_LOSS_PCT",
            # 风控参数
            "ACCOUNT_EQUITY_USDT",
            "RISK_BUDGET_PCT",
            "MAX_DRAWDOWN_PCT",
            "MAX_CONCURRENT_POSITIONS",
            "MIN_ORDER_USDT",
            # AI参数
            "AI_ENABLED",
            "AI_WEIGHT",
            "AI_LR",
            "AI_MIN_SAMPLES",
        ),
        )
        db_symbols = _parse_symbols(m.get("SYMBOLS"))
        # symbols 回退：优先数据库，其次 settings，最后默认空
        symbols = db_symbols if db_symbols else (tuple(settings.symbols) if settings else tuple())
        symbols_from_db = bool(db_symbols)

        return cls(
            symbols=symbols,
            symbols_from_db=symbols_from_db,
            halt_trading=_parse_bool(m.get("HALT_TRADING"), default=False),
            emergency_exit=_parse_bool(m.get("EMERGENCY_EXIT"), default=False),
            # 停止订单配置：仅支持数据库配置，无 .env 回退
            use_protective_stop_order=_parse_bool(m.get("USE_PROTECTIVE_STOP_ORDER"), default=True),
            stop_order_poll_seconds=_parse_int(m.get("STOP_ORDER_POLL_SECONDS"), default=10),
            stop_arm_max_retries=_parse_int(m.get("STOP_ARM_MAX_RETRIES"), default=3),
            stop_arm_backoff_base_seconds=_parse_float(m.get("STOP_ARM_BACKOFF_BASE_SECONDS"), default=0.5),
            stop_rearm_max_attempts=_parse_int(m.get("STOP_REARM_MAX_ATTEMPTS"), default=2),
            stop_rearm_cooldown_seconds=_parse_int(m.get("STOP_REARM_COOLDOWN_SECONDS"), default=60),
            # 策略参数：仅支持数据库配置，无 .env 回退
            setup_b_adx_min=_parse_float(m.get("SETUP_B_ADX_MIN"), default=20.0),
            setup_b_vol_ratio_min=_parse_float(m.get("SETUP_B_VOL_RATIO_MIN"), default=1.5),
            setup_b_ai_score_min=_parse_float(m.get("SETUP_B_AI_SCORE_MIN"), default=55.0),
            hard_stop_loss_pct=_parse_float(m.get("HARD_STOP_LOSS_PCT"), default=0.03),
            # 风控参数：仅支持数据库配置，无 .env 回退
            account_equity_usdt=_parse_float(m.get("ACCOUNT_EQUITY_USDT"), default=500.0),
            risk_budget_pct=_parse_float(m.get("RISK_BUDGET_PCT"), default=0.03),
            max_drawdown_pct=_parse_float(m.get("MAX_DRAWDOWN_PCT"), default=0.15),
            max_concurrent_positions=_parse_int(m.get("MAX_CONCURRENT_POSITIONS"), default=3),
            min_order_usdt=_parse_float(m.get("MIN_ORDER_USDT"), default=50.0),
            # AI参数：仅支持数据库配置，无 .env 回退
            ai_enabled=_parse_bool(m.get("AI_ENABLED"), default=True),
            ai_weight=_parse_float(m.get("AI_WEIGHT"), default=0.35),
            ai_lr=_parse_float(m.get("AI_LR"), default=0.05),
            ai_min_samples=_parse_int(m.get("AI_MIN_SAMPLES"), default=50),
            last_refresh_ms=int(time.time() * 1000),
        )

    def refresh(self, db: PostgreSQL, settings: Settings = None) -> Dict[str, Any]:
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
            self.setup_b_adx_min,
            self.setup_b_vol_ratio_min,
            self.setup_b_ai_score_min,
            self.hard_stop_loss_pct,
            self.account_equity_usdt,
            self.risk_budget_pct,
            self.max_drawdown_pct,
            self.max_concurrent_positions,
            self.min_order_usdt,
            self.ai_enabled,
            self.ai_weight,
            self.ai_lr,
            self.ai_min_samples,
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
            "SETUP_B_ADX_MIN",
            "SETUP_B_VOL_RATIO_MIN",
            "SETUP_B_AI_SCORE_MIN",
            "HARD_STOP_LOSS_PCT",
            "ACCOUNT_EQUITY_USDT",
            "RISK_BUDGET_PCT",
            "MAX_DRAWDOWN_PCT",
            "MAX_CONCURRENT_POSITIONS",
            "MIN_ORDER_USDT",
            "AI_ENABLED",
            "AI_WEIGHT",
            "AI_LR",
            "AI_MIN_SAMPLES",
        ),
        )
        db_symbols = _parse_symbols(m.get("SYMBOLS"))
        # symbols 回退：优先数据库，其次 settings（如果提供），最后默认空
        self.symbols = db_symbols if db_symbols else (tuple(settings.symbols) if settings else tuple())
        self.symbols_from_db = bool(db_symbols)
        self.halt_trading = _parse_bool(m.get("HALT_TRADING"), default=False)
        self.emergency_exit = _parse_bool(m.get("EMERGENCY_EXIT"), default=False)
        # 停止订单配置：仅支持数据库配置，无 .env 回退
        self.use_protective_stop_order = _parse_bool(m.get("USE_PROTECTIVE_STOP_ORDER"), default=True)
        self.stop_order_poll_seconds = _parse_int(m.get("STOP_ORDER_POLL_SECONDS"), default=10)
        self.stop_arm_max_retries = _parse_int(m.get("STOP_ARM_MAX_RETRIES"), default=3)
        self.stop_arm_backoff_base_seconds = _parse_float(m.get("STOP_ARM_BACKOFF_BASE_SECONDS"), default=0.5)
        self.stop_rearm_max_attempts = _parse_int(m.get("STOP_REARM_MAX_ATTEMPTS"), default=2)
        self.stop_rearm_cooldown_seconds = _parse_int(m.get("STOP_REARM_COOLDOWN_SECONDS"), default=60)
        # 策略参数：仅支持数据库配置，无 .env 回退
        self.setup_b_adx_min = _parse_float(m.get("SETUP_B_ADX_MIN"), default=20.0)
        self.setup_b_vol_ratio_min = _parse_float(m.get("SETUP_B_VOL_RATIO_MIN"), default=1.5)
        self.setup_b_ai_score_min = _parse_float(m.get("SETUP_B_AI_SCORE_MIN"), default=55.0)
        self.hard_stop_loss_pct = _parse_float(m.get("HARD_STOP_LOSS_PCT"), default=0.03)
        # 风控参数：仅支持数据库配置，无 .env 回退
        self.account_equity_usdt = _parse_float(m.get("ACCOUNT_EQUITY_USDT"), default=500.0)
        self.risk_budget_pct = _parse_float(m.get("RISK_BUDGET_PCT"), default=0.03)
        self.max_drawdown_pct = _parse_float(m.get("MAX_DRAWDOWN_PCT"), default=0.15)
        self.max_concurrent_positions = _parse_int(m.get("MAX_CONCURRENT_POSITIONS"), default=3)
        self.min_order_usdt = _parse_float(m.get("MIN_ORDER_USDT"), default=50.0)
        # AI参数：仅支持数据库配置，无 .env 回退
        self.ai_enabled = _parse_bool(m.get("AI_ENABLED"), default=True)
        self.ai_weight = _parse_float(m.get("AI_WEIGHT"), default=0.35)
        self.ai_lr = _parse_float(m.get("AI_LR"), default=0.05)
        self.ai_min_samples = _parse_int(m.get("AI_MIN_SAMPLES"), default=50)
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
            self.setup_b_adx_min,
            self.setup_b_vol_ratio_min,
            self.setup_b_ai_score_min,
            self.hard_stop_loss_pct,
            self.account_equity_usdt,
            self.risk_budget_pct,
            self.max_drawdown_pct,
            self.max_concurrent_positions,
            self.min_order_usdt,
            self.ai_enabled,
            self.ai_weight,
            self.ai_lr,
            self.ai_min_samples,
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
        # 策略参数变更检测
        if before[10] != after[10]:
            changes["setup_b_adx_min"] = self.setup_b_adx_min
        if before[11] != after[11]:
            changes["setup_b_vol_ratio_min"] = self.setup_b_vol_ratio_min
        if before[12] != after[12]:
            changes["setup_b_ai_score_min"] = self.setup_b_ai_score_min
        if before[13] != after[13]:
            changes["hard_stop_loss_pct"] = self.hard_stop_loss_pct
        # 风控参数变更检测
        if before[14] != after[14]:
            changes["account_equity_usdt"] = self.account_equity_usdt
        if before[15] != after[15]:
            changes["risk_budget_pct"] = self.risk_budget_pct
        if before[16] != after[16]:
            changes["max_drawdown_pct"] = self.max_drawdown_pct
        if before[17] != after[17]:
            changes["max_concurrent_positions"] = self.max_concurrent_positions
        if before[18] != after[18]:
            changes["min_order_usdt"] = self.min_order_usdt
        # AI参数变更检测
        if before[19] != after[19]:
            changes["ai_enabled"] = self.ai_enabled
        if before[20] != after[20]:
            changes["ai_weight"] = self.ai_weight
        if before[21] != after[21]:
            changes["ai_lr"] = self.ai_lr
        if before[22] != after[22]:
            changes["ai_min_samples"] = self.ai_min_samples
        return changes
