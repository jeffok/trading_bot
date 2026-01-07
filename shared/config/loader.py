"""Configuration loader.

Design goals:
- All services share a single Settings model.
- Environment variables are the source of truth (12-factor style).
- Keep configuration explicit, typed, and self-documented.

NOTE:
- Scheduler uses Asia/Hong_Kong time; DB timestamps are stored in UTC.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=False)


def _env_first(*names: str, default: str = "") -> str:
    """按顺序从多个 env key 取值，取到第一个非空的。"""
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def _parse_symbols_env() -> tuple[str, ...]:
    """Parse SYMBOLS env (comma/space separated). Fallback to SYMBOL.

    Examples:
      SYMBOLS="BTCUSDT,ETHUSDT" -> ("BTCUSDT","ETHUSDT")
      SYMBOL="BTCUSDT" -> ("BTCUSDT",)
    """
    raw = _env_first("SYMBOLS", default="")
    if not raw:
        raw = _env_first("SYMBOL", default="BTCUSDT")
    # allow comma or whitespace separated
    parts = []
    for token in re.split(r"[\s,]+", raw.strip()):
        t = token.strip().upper()
        if t:
            parts.append(t)
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return tuple(uniq) if uniq else ("BTCUSDT",)


def _parse_csv_env(name: str, *, fallback: str = "", upper: bool = False) -> tuple[str, ...]:
    """Parse comma/space separated env into tuple.

    If env is empty, use fallback.
    """
    raw = os.getenv(name, "")
    raw = str(raw).strip() if raw is not None else ""
    if not raw:
        raw = str(fallback).strip()
    if not raw:
        return tuple()
    parts = [p.strip() for p in re.split(r"[\s,]+", raw) if p.strip()]
    if upper:
        parts = [p.upper() for p in parts]
    return tuple(parts)




@dataclass(frozen=True)
class Settings:
    # ✅ 兼容字段：有些地方用 env，有些地方用 app_env
    # 优先 ENV，其次 APP_ENV
    env: str = _env_first("ENV", "APP_ENV", default="dev")
    app_env: str = _env_first("APP_ENV", "ENV", default="dev")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # 交易所
    exchange: str = os.getenv("EXCHANGE", "paper").lower()
    exchange_category: str = os.getenv("EXCHANGE_CATEGORY", "linear").lower()
    futures_leverage: int = int(os.getenv("FUTURES_LEVERAGE", "3"))
    bybit_position_idx: int = int(os.getenv("BYBIT_POSITION_IDX", "0"))
    symbol: str = os.getenv("SYMBOL", "BTCUSDT").upper()
    # 多交易对池（优先 SYMBOLS，其次 SYMBOL）
    symbols: tuple[str, ...] = field(default_factory=_parse_symbols_env)

    interval_minutes: int = int(os.getenv("INTERVAL_MINUTES", "15"))
    strategy_tick_seconds: int = int(os.getenv("STRATEGY_TICK_SECONDS", "900"))
    hard_stop_loss_pct: float = float(os.getenv("HARD_STOP_LOSS_PCT", "0.03"))

    # 交易与风控（MVP 默认）
    max_concurrent_positions: int = int(os.getenv('MAX_CONCURRENT_POSITIONS', '3'))
    # 每单最小保证金（USDT）- 由策略侧反推 qty；名义价值通常为保证金*杠杆
    min_order_usdt: float = float(os.getenv('MIN_ORDER_USDT', '50'))
    # 根据评分自动选择杠杆范围
    auto_leverage_min: int = int(os.getenv('AUTO_LEVERAGE_MIN', '10'))
    auto_leverage_max: int = int(os.getenv('AUTO_LEVERAGE_MAX', '20'))

    # Take profit (optional): if enabled, profitable exits are labeled as TAKE_PROFIT
    take_profit_reason_on_positive_pnl: bool = os.getenv("TAKE_PROFIT_REASON_ON_POSITIVE_PNL", "true").strip().lower() in ("1","true","yes","y")

    # AI (online learning)
    ai_enabled: bool = os.getenv("AI_ENABLED", "true").strip().lower() in ("1","true","yes","y")
    ai_weight: float = float(os.getenv("AI_WEIGHT", "0.35"))  # 0..1
    ai_lr: float = float(os.getenv("AI_LR", "0.05"))
    ai_l2: float = float(os.getenv("AI_L2", "0.000001"))
    ai_min_samples: int = int(os.getenv("AI_MIN_SAMPLES", "50"))
    ai_model_key: str = os.getenv("AI_MODEL_KEY", "AI_MODEL_V1")

    # Drills / tests: run one cycle then exit
    run_once: bool = os.getenv("RUN_ONCE", "false").strip().lower() in ("1","true","yes","y")


    admin_token: str = os.getenv("ADMIN_TOKEN", "change_me")

    # 支持多 token（逗号/空格分隔），用于密钥轮换；若未设置 ADMIN_TOKENS，则回退 ADMIN_TOKEN
    admin_tokens: tuple[str, ...] = _parse_csv_env(
        "ADMIN_TOKENS",
        fallback=os.getenv("ADMIN_TOKEN", "change_me"),
        upper=False,
    )

    # 管理接口 IP 白名单（CIDR 或单 IP，逗号/空格分隔）。为空则不启用。
    admin_ip_allowlist: tuple[str, ...] = _parse_csv_env("ADMIN_IP_ALLOWLIST", fallback="", upper=False)

    # 高危操作二次确认（可选）：若开启，则 /admin/emergency_exit 与 CLI emergency-exit 需要 confirm_code
    admin_confirm_required: bool = os.getenv("ADMIN_CONFIRM_REQUIRED", "false").strip().lower() in ("1","true","yes","y")
    admin_confirm_code: str = os.getenv("ADMIN_CONFIRM_CODE", "")

    # Leader election (HA)：多实例时仅 leader 执行同步/交易，followers 仅心跳与指标
    leader_election_enabled: bool = os.getenv("LEADER_ELECTION_ENABLED", "true").strip().lower() in ("1","true","yes","y")
    leader_key_prefix: str = os.getenv("LEADER_KEY_PREFIX", "leader")
    leader_ttl_seconds: int = int(os.getenv("LEADER_TTL_SECONDS", "30"))
    leader_renew_interval_seconds: int = int(os.getenv("LEADER_RENEW_INTERVAL_SECONDS", "10"))
    leader_follower_sleep_seconds: int = int(os.getenv("LEADER_FOLLOWER_SLEEP_SECONDS", "2"))

    # DB / Redis（外部）
    db_host: str = os.getenv("DB_HOST", "mariadb")
    db_port: int = int(os.getenv("DB_PORT", "3306"))
    db_user: str = os.getenv("DB_USER", "alpha")
    db_pass: str = os.getenv("DB_PASS", "alpha_pass")
    db_name: str = os.getenv("DB_NAME", "alpha_sniper")

    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


    # Observability / runtime identity
    # METRICS_PORT=0 means "auto" (service decides default) or disabled if service doesn't expose a metrics port.
    metrics_port: int = int(os.getenv("METRICS_PORT", "0"))
    # Optional: force instance id (otherwise service will use hostname:pid)
    instance_id: str = os.getenv("INSTANCE_ID", "")
    heartbeat_interval_seconds: int = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30"))


    # Binance USDT-M Futures
    binance_base_url: str = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    binance_recv_window: int = int(os.getenv("BINANCE_RECV_WINDOW", "5000"))

    # Bybit Linear
    bybit_base_url: str = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
    bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_api_secret: str = os.getenv("BYBIT_API_SECRET", "")
    bybit_recv_window: int = int(os.getenv("BYBIT_RECV_WINDOW", "5000"))

    # paper（如果你不用 paper，这些不会影响）
    paper_starting_usdt: float = float(os.getenv("PAPER_STARTING_USDT", "1000"))
    paper_fee_pct: float = float(os.getenv("PAPER_FEE_PCT", "0.0004"))

    def is_telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)
