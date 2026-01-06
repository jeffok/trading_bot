
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
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=False)

@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    exchange: str = os.getenv("EXCHANGE", "paper").lower()
    exchange_category: str = os.getenv("EXCHANGE_CATEGORY", "spot").lower()
    symbol: str = os.getenv("SYMBOL", "BTCUSDT").upper()
    interval_minutes: int = int(os.getenv("INTERVAL_MINUTES", "15"))
    strategy_tick_seconds: int = int(os.getenv("STRATEGY_TICK_SECONDS", "900"))
    hard_stop_loss_pct: float = float(os.getenv("HARD_STOP_LOSS_PCT", "0.03"))

    admin_token: str = os.getenv("ADMIN_TOKEN", "change_me")

    db_host: str = os.getenv("DB_HOST", "mariadb")
    db_port: int = int(os.getenv("DB_PORT", "3306"))
    db_user: str = os.getenv("DB_USER", "alpha")
    db_pass: str = os.getenv("DB_PASS", "alpha_pass")
    db_name: str = os.getenv("DB_NAME", "alpha_sniper")

    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    binance_base_url: str = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    binance_recv_window: int = int(os.getenv("BINANCE_RECV_WINDOW", "5000"))

    bybit_base_url: str = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
    bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_api_secret: str = os.getenv("BYBIT_API_SECRET", "")
    bybit_recv_window: int = int(os.getenv("BYBIT_RECV_WINDOW", "5000"))

    paper_starting_usdt: float = float(os.getenv("PAPER_STARTING_USDT", "1000"))
    paper_fee_pct: float = float(os.getenv("PAPER_FEE_PCT", "0.0004"))

    def is_telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)
