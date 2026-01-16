from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from shared.config import Settings
from shared.db import PostgreSQL, migrate
from services.data_syncer.main import compute_features_for_bars


def main() -> None:
    settings = Settings()
    db = PostgreSQL(settings.postgres_url)
    migrate(db, migrations_dir="migrations")

    symbol = (settings.symbols[0] if settings.symbols else settings.symbol).upper()
    interval = int(settings.interval_minutes)

    # Generate synthetic 15m bars for last ~3 days (288 bars) with gentle uptrend then pullback
    n = int(os.getenv("SEED_BARS", "260"))
    now = datetime.now(timezone.utc)
    # align to interval boundary
    base_ts = int(now.timestamp() // (interval * 60) * (interval * 60))
    start_ts = base_ts - n * interval * 60

    bars: List[Dict[str, Any]] = []
    price = float(os.getenv("SEED_START_PRICE", "40000"))
    for i in range(n):
        t0 = start_ts + i * interval * 60
        open_time_ms = int(t0 * 1000)
        close_time_ms = int((t0 + interval * 60 - 1) * 1000)

        # shape: first 70% uptrend, last 30% mild pullback/sideways
        if i < int(n * 0.7):
            price = price * 1.0015
        else:
            price = price * 0.9995

        o = price * 0.999
        c = price
        h = max(o, c) * 1.001
        l = min(o, c) * 0.999
        v = 100 + (i % 20) * 2

        bars.append(
            dict(
                open_time_ms=open_time_ms,
                close_time_ms=close_time_ms,
                open_price=o,
                high_price=h,
                low_price=l,
                close_price=c,
                volume=v,
            )
        )

    # Compute cache features for these bars
    feats = compute_features_for_bars(
        [
            {
                "open_time_ms": b["open_time_ms"],
                "close_price": float(b["close_price"]),
                "volume": float(b["volume"]),
                "high_price": float(b["high_price"]),
                "low_price": float(b["low_price"]),
            }
            for b in bars
        ],
        ema_fast_period=int(os.getenv("EMA_FAST", "12")),
        ema_slow_period=int(os.getenv("EMA_SLOW", "26")),
        rsi_period=int(os.getenv("RSI_PERIOD", "14")),
    )

    # Insert into DB (idempotent by PK)
    with db.tx() as cur:
        for b, c in zip(bars, feats):
            cur.execute(
                """
                INSERT INTO market_data(symbol, interval_minutes, open_time_ms, close_time_ms, open_price, high_price, low_price, close_price, volume)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (symbol, interval_minutes, open_time_ms) DO NOTHING
                """,
                (
                    symbol,
                    interval,
                    int(b["open_time_ms"]),
                    int(b["close_time_ms"]),
                    float(b["open_price"]),
                    float(b["high_price"]),
                    float(b["low_price"]),
                    float(b["close_price"]),
                    float(b["volume"]),
                ),
            )
            cur.execute(
                """
                INSERT INTO market_data_cache(symbol, interval_minutes, open_time_ms, feature_version, ema_fast, ema_slow, rsi, features_json, created_at)
                VALUES (%s,%s,%s,1,%s,%s,%s,%s, CURRENT_TIMESTAMP)
                ON CONFLICT (symbol, interval_minutes, open_time_ms, feature_version) DO NOTHING
                """,
                (
                    symbol,
                    interval,
                    int(b["open_time_ms"]),
                    c.get("ema_fast"),
                    c.get("ema_slow"),
                    c.get("rsi"),
                    c.get("features_json"),
                ),
            )

    print(f"seed ok: symbol={symbol} interval={interval} bars={n}")


if __name__ == "__main__":
    main()
