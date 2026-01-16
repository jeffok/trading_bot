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


def _build_postgres_url() -> str:
    """构建PostgreSQL连接URL（支持从旧MySQL参数自动转换）"""
    # 优先使用POSTGRES_URL
    postgres_url = os.getenv("POSTGRES_URL", "").strip()
    if postgres_url:
        return postgres_url
    
    # 向后兼容：从旧的MySQL参数构建PostgreSQL URL
    db_host = os.getenv("DB_HOST", "").strip()
    db_port = os.getenv("DB_PORT", "5432").strip()
    db_user = os.getenv("DB_USER", "").strip()
    db_pass = os.getenv("DB_PASS", "").strip()
    db_name = os.getenv("DB_NAME", "").strip()
    
    if db_host and db_user and db_pass and db_name:
        return f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    # 默认值
    return "postgresql://postgres:password@postgres:5432/trading_bot"




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

    # Setup B (V8.3)
    setup_b_adx_min: float = float(os.getenv("SETUP_B_ADX_MIN", "20"))
    setup_b_vol_ratio_min: float = float(os.getenv("SETUP_B_VOL_RATIO_MIN", "1.5"))
    setup_b_ai_score_min: float = float(os.getenv("SETUP_B_AI_SCORE_MIN", "55"))

    # Risk budget / circuit breaker (V8.3)
    account_equity_usdt: float = float(os.getenv("ACCOUNT_EQUITY_USDT", "500"))
    risk_budget_pct: float = float(os.getenv("RISK_BUDGET_PCT", "0.03"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.15"))
    circuit_window_seconds: int = int(os.getenv("CIRCUIT_WINDOW_SECONDS", "600"))
    circuit_rate_limit_threshold: int = int(os.getenv("CIRCUIT_RATE_LIMIT_THRESHOLD", "8"))
    circuit_failure_threshold: int = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "6"))
    btc_symbol: str = os.getenv("BTC_SYMBOL", "BTCUSDT").upper()

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
    ai_model_impl: str = os.getenv("AI_MODEL_IMPL", "online_lr").strip().lower()  # online_lr | sgd_compat

    # Drills / tests: run one cycle then exit
    run_once: bool = os.getenv("RUN_ONCE", "false").strip().lower() in ("1","true","yes","y")

    # Runtime config refresh
    runtime_config_refresh_seconds: int = int(os.getenv("RUNTIME_CONFIG_REFRESH_SECONDS", "30"))
    use_protective_stop_order: bool = os.getenv("USE_PROTECTIVE_STOP_ORDER", "true").strip().lower() in ("1","true","yes","y")
    stop_order_poll_seconds: int = int(os.getenv("STOP_ORDER_POLL_SECONDS", "10"))
    
    # Data syncer optimization
    data_sync_loop_interval_seconds: float = float(os.getenv("DATA_SYNC_LOOP_INTERVAL_SECONDS", "30"))
    data_sync_symbol_delay_seconds: float = float(os.getenv("DATA_SYNC_SYMBOL_DELAY_SECONDS", "0.5"))

    # Control commands polling (V8.3): poll NEW commands every 1~3 seconds
    control_poll_seconds: float = float(os.getenv("CONTROL_POLL_SECONDS", "2"))

    # Feature cache versioning (V8.3)
    feature_version: int = int(os.getenv("FEATURE_VERSION", "1"))

    # Tick budget (V8.3): each tick should finish within ~10s
    tick_budget_seconds: float = float(os.getenv("TICK_BUDGET_SECONDS", "10"))

    # Position snapshots (V8.3): write snapshot every 5 minutes
    position_snapshot_interval_seconds: int = int(os.getenv("POSITION_SNAPSHOT_INTERVAL_SECONDS", "300"))

    # Distributed trade lock TTL (V8.3)
    trade_lock_ttl_seconds: int = int(os.getenv("TRADE_LOCK_TTL_SECONDS", "30"))

    # B2: protective stop abnormal handling
    stop_arm_max_retries: int = int(os.getenv("STOP_ARM_MAX_RETRIES", "3"))
    stop_arm_backoff_base_seconds: float = float(os.getenv("STOP_ARM_BACKOFF_BASE_SECONDS", "0.5"))
    stop_rearm_max_attempts: int = int(os.getenv("STOP_REARM_MAX_ATTEMPTS", "2"))
    stop_rearm_cooldown_seconds: int = int(os.getenv("STOP_REARM_COOLDOWN_SECONDS", "60"))

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
    # PostgreSQL连接URL（格式：postgresql://user:password@host:port/dbname）
    postgres_url: str = field(default_factory=_build_postgres_url)
    
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # Telegram
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Data lag alerts (V8.3): send Telegram alert when cache lag exceeds threshold
    market_data_lag_alert_seconds: float = float(os.getenv("MARKET_DATA_LAG_ALERT_SECONDS", "120"))
    market_data_lag_alert_cooldown_seconds: float = float(os.getenv("MARKET_DATA_LAG_ALERT_COOLDOWN_SECONDS", "300"))


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


ALLOWED_EXCHANGES: set[str] = {"binance", "bybit", "paper"}


def load_settings() -> Settings:
    """Create Settings with basic validation.

    Notes:
    - 实盘运行时只会选择一个交易所（EXCHANGE=binance/bybit），不会同时连接多个交易所。
      但仓库会同时包含多交易所适配层，供你通过配置切换。
    """
    s = Settings()
    ex = (s.exchange or "").strip().lower()
    if ex not in ALLOWED_EXCHANGES:
        raise ValueError(
            f"Invalid EXCHANGE={s.exchange!r}. Allowed: {', '.join(sorted(ALLOWED_EXCHANGES))}"
        )
    return s