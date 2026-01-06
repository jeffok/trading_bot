from __future__ import annotations

import os
import time
from pathlib import Path

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.exchange import make_exchange
from shared.logging import get_logger, new_trace_id
from shared.redis import distributed_lock, redis_client
from shared.telemetry import Metrics, Telegram
from shared.models.market_cache import MarketDataCacheRepo
from shared.models.market_repo import MarketDataRepo
from shared.models.service_status_repo import ServiceStatusRepo
from shared.strategy.indicators import compute_indicators_for_series

SERVICE = "data-syncer"
VERSION = "0.1.0"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))


def tg_alert(telegram: Telegram, *, level: str, event: str, title: str, trace_id: str, exchange: str, symbol: str, summary_extra: dict, payload_extra: dict) -> None:
    summary_kv = {
        "level": level,
        "event": event,
        "service": "行情同步",
        "trace_id": trace_id,
        "exchange": exchange,  # 保持默认
        "symbol": symbol,      # 保持默认
        **(summary_extra or {}),
    }
    payload = {
        "level": level,
        "event": event,
        "service": "data-syncer",
        "trace_id": trace_id,
        "exchange": exchange,
        "symbol": symbol,
        **(payload_extra or {}),
    }
    telegram.send_alert_zh(title=title, summary_kv=summary_kv, payload=payload)


def main() -> None:
    settings = Settings()
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    metrics = Metrics(service=SERVICE, instance=settings.instance_id)

    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, Path("/app/migrations"))

    r = redis_client(settings.redis_url)

    market_repo = MarketDataRepo(db)
    cache_repo = MarketDataCacheRepo(db)
    svc_repo = ServiceStatusRepo(db)

    ex = make_exchange(settings)

    while True:
        trace_id = new_trace_id("sync")
        try:
            with distributed_lock(r, key=f"lock:{SERVICE}:{settings.symbol}", ttl_seconds=25):
                svc_repo.heartbeat(service_name=SERVICE, instance_id=settings.instance_id)

                candles = ex.fetch_klines(symbol=settings.symbol, interval_minutes=settings.interval_minutes, limit=300)
                if not candles:
                    time.sleep(settings.data_sync_interval_seconds)
                    continue

                market_repo.upsert_candles(symbol=settings.symbol, interval_minutes=settings.interval_minutes, candles=candles)

                series = market_repo.get_latest_series(symbol=settings.symbol, interval_minutes=settings.interval_minutes, limit=300)
                ind = compute_indicators_for_series(series)

                cache_repo.upsert_latest(symbol=settings.symbol, interval_minutes=settings.interval_minutes, snapshot=ind)

                ex.update_last_price(symbol=settings.symbol, price=float(ind["close_price"]))

                metrics.counter("sync_ok_total").inc()
                time.sleep(settings.data_sync_interval_seconds)

        except Exception as e:
            logger.exception("data-syncer loop error trace_id=%s", trace_id)
            tg_alert(
                telegram,
                level="ERROR",
                event="DATA_SYNC_ERROR",
                title=f"❌ 行情同步错误 {settings.symbol}",
                trace_id=trace_id,
                exchange=settings.exchange,
                symbol=settings.symbol,
                summary_extra={"错误": str(e)[:200]},
                payload_extra={"reason_code": "DATA_SYNC", "error": str(e)},
            )
            time.sleep(2.0)


if __name__ == "__main__":
    main()
