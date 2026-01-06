
from __future__ import annotations

import os
import time
from pathlib import Path

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.exchange import make_exchange
from shared.logging import get_logger, new_trace_id
from shared.telemetry import Metrics, Telegram
from shared.domain.time import now_ms

SERVICE = "data-syncer"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))

def compute_ema(prev: float, price: float, period: int) -> float:
    k = 2 / (period + 1)
    return price * k + prev * (1 - k)

def compute_rsi(closes, period: int = 14):
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i-1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def upsert_heartbeat(db: MariaDB, instance_id: str, status: dict):
    db.execute(
        """
        INSERT INTO service_status(service_name, instance_id, last_heartbeat, status_json)
        VALUES (%s,%s, CURRENT_TIMESTAMP, %s)
        ON DUPLICATE KEY UPDATE last_heartbeat=CURRENT_TIMESTAMP, status_json=VALUES(status_json)
        """,
        (SERVICE, instance_id, __import__("json").dumps(status, ensure_ascii=False)),
    )

def main():
    settings = Settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, Path("/app/migrations"))

    metrics = Metrics(SERVICE)
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    ex = make_exchange(settings, metrics=metrics, service_name=SERVICE)

    instance_id = f"{SERVICE}-{os.getpid()}"
    ema_fast = None
    ema_slow = None

    last_cache = db.fetch_one(
        """
        SELECT ema_fast, ema_slow FROM market_data_cache
        WHERE symbol=%s AND interval_minutes=%s
        ORDER BY open_time_ms DESC LIMIT 1
        """,
        (settings.symbol, settings.interval_minutes),
    )
    if last_cache:
        ema_fast = float(last_cache["ema_fast"]) if last_cache["ema_fast"] is not None else None
        ema_slow = float(last_cache["ema_slow"]) if last_cache["ema_slow"] is not None else None

    while True:
        trace_id = new_trace_id("sync")
        try:
            last = db.fetch_one(
                """
                SELECT open_time_ms FROM market_data
                WHERE symbol=%s AND interval_minutes=%s
                ORDER BY open_time_ms DESC LIMIT 1
                """,
                (settings.symbol, settings.interval_minutes),
            )
            start_ms = int(last["open_time_ms"]) + settings.interval_minutes * 60_000 if last else None

            klines = ex.fetch_klines(symbol=settings.symbol, interval_minutes=settings.interval_minutes, start_ms=start_ms, limit=1000)

            if not klines:
                upsert_heartbeat(db, instance_id, {"trace_id": trace_id, "status": "NO_DATA"})
                time.sleep(10)
                continue

            recent = db.fetch_all(
                """
                SELECT close_price FROM market_data
                WHERE symbol=%s AND interval_minutes=%s
                ORDER BY open_time_ms DESC LIMIT 100
                """,
                (settings.symbol, settings.interval_minutes),
            )
            closes = [float(r["close_price"]) for r in reversed(recent)]

            md_rows = []
            cache_rows = []
            for k in klines:
                md_rows.append((settings.symbol, settings.interval_minutes, k.open_time_ms, k.close_time_ms, k.open, k.high, k.low, k.close, k.volume))
                closes.append(float(k.close))
                if ema_fast is None:
                    ema_fast = float(k.close)
                else:
                    ema_fast = compute_ema(ema_fast, float(k.close), 7)
                if ema_slow is None:
                    ema_slow = float(k.close)
                else:
                    ema_slow = compute_ema(ema_slow, float(k.close), 25)

                rsi = compute_rsi(closes, 14)
                cache_rows.append((settings.symbol, settings.interval_minutes, k.open_time_ms, ema_fast, ema_slow, rsi))

            with db.tx() as cur:
                cur.executemany(
                    """
                    INSERT INTO market_data(
                      symbol, interval_minutes, open_time_ms, close_time_ms,
                      open_price, high_price, low_price, close_price, volume
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      close_time_ms=VALUES(close_time_ms),
                      open_price=VALUES(open_price),
                      high_price=VALUES(high_price),
                      low_price=VALUES(low_price),
                      close_price=VALUES(close_price),
                      volume=VALUES(volume)
                    """,
                    md_rows,
                )
                cur.executemany(
                    """
                    INSERT INTO market_data_cache(symbol, interval_minutes, open_time_ms, ema_fast, ema_slow, rsi)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      ema_fast=VALUES(ema_fast),
                      ema_slow=VALUES(ema_slow),
                      rsi=VALUES(rsi)
                    """,
                    cache_rows,
                )

            if hasattr(ex, "update_last_price"):
                ex.update_last_price(settings.symbol, float(klines[-1].close))

            lag = max(0, now_ms() - klines[-1].close_time_ms)
            metrics.data_sync_lag_ms.labels(SERVICE, settings.symbol, str(settings.interval_minutes)).set(lag)
            upsert_heartbeat(db, instance_id, {"trace_id": trace_id, "status": "OK", "inserted": len(md_rows), "lag_ms": lag})
        except Exception as e:
            telegram.send(f"[{SERVICE}] ERROR trace_id={trace_id} err={e}")
            upsert_heartbeat(db, instance_id, {"trace_id": trace_id, "status": "ERROR", "error": str(e)[:400]})
            time.sleep(5)

        time.sleep(10)

if __name__ == "__main__":
    main()
