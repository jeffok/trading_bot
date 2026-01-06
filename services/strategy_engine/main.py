
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.exchange import make_exchange
from shared.logging import get_logger, new_trace_id
from shared.redis import distributed_lock, redis_client
from shared.telemetry import Metrics, Telegram
from shared.domain.enums import OrderEventType, ReasonCode, Side
from shared.domain.events import append_order_event
from shared.domain.time import next_tick_sleep_seconds

SERVICE = "strategy-engine"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))

def get_flag(db: MariaDB, key: str, default: str = "false") -> str:
    row = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
    return (row["value"] if row else default).strip().lower()

def set_flag(db: MariaDB, key: str, value: str) -> None:
    db.execute("INSERT INTO system_config(`key`,`value`) VALUES (%s,%s) ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)", (key, value))

def latest_cache(db: MariaDB, symbol: str, interval_minutes: int):
    return db.fetch_one(
        """
        SELECT m.open_time_ms, m.close_price, c.ema_fast, c.ema_slow, c.rsi
        FROM market_data m
        LEFT JOIN market_data_cache c
          ON c.symbol=m.symbol AND c.interval_minutes=m.interval_minutes AND c.open_time_ms=m.open_time_ms
        WHERE m.symbol=%s AND m.interval_minutes=%s
        ORDER BY m.open_time_ms DESC
        LIMIT 1
        """,
        (symbol, interval_minutes),
    )

def get_position(db: MariaDB, symbol: str):
    return db.fetch_one(
        """
        SELECT base_qty, avg_entry_price, meta_json
        FROM position_snapshots
        WHERE symbol=%s
        ORDER BY id DESC LIMIT 1
        """,
        (symbol,),
    )

def save_position(db: MariaDB, symbol: str, base_qty: float, avg_entry_price: Optional[float], meta: dict) -> None:
    db.execute(
        """
        INSERT INTO position_snapshots(symbol, base_qty, avg_entry_price, meta_json)
        VALUES (%s,%s,%s,%s)
        """,
        (symbol, float(base_qty), float(avg_entry_price) if avg_entry_price is not None else None, json.dumps(meta, ensure_ascii=False)),
    )

def setup_b_signal(latest: dict) -> Optional[str]:
    ema_fast = latest.get("ema_fast")
    ema_slow = latest.get("ema_slow")
    rsi = latest.get("rsi")
    if ema_fast is None or ema_slow is None:
        return None
    if float(ema_fast) > float(ema_slow) and (rsi is None or float(rsi) < 70):
        return "BUY"
    if float(ema_fast) < float(ema_slow):
        return "SELL"
    return None

def main():
    settings = Settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, Path("/app/migrations"))

    metrics = Metrics(SERVICE)
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    r = redis_client(settings.redis_url)
    ex = make_exchange(settings, metrics=metrics, service_name=SERVICE)

    while True:
        time.sleep(next_tick_sleep_seconds(settings.strategy_tick_seconds))
        trace_id = new_trace_id("tick")

        lock_key = f"lock:tick:{settings.exchange}:{settings.symbol}:{int(time.time() // settings.strategy_tick_seconds)}"
        with distributed_lock(r, lock_key, ttl_ms=int(settings.strategy_tick_seconds * 1000 * 0.9)) as acquired:
            if not acquired:
                continue

            try:
                if get_flag(db, "HALT_TRADING", "false") == "true":
                    telegram.send(f"[HALT] skip tick trace_id={trace_id} symbol={settings.symbol}")
                    continue

                latest = latest_cache(db, settings.symbol, settings.interval_minutes)
                if not latest:
                    telegram.send(f"[{SERVICE}] no market data trace_id={trace_id}")
                    continue

                last_price = float(latest["close_price"])
                if hasattr(ex, "update_last_price"):
                    ex.update_last_price(settings.symbol, last_price)

                pos = get_position(db, settings.symbol)
                base_qty = float(pos["base_qty"]) if pos else 0.0
                avg_entry = float(pos["avg_entry_price"]) if pos and pos["avg_entry_price"] is not None else None

                # Emergency exit
                if get_flag(db, "EMERGENCY_EXIT", "false") == "true":
                    if base_qty > 0:
                        client_order_id = f"exit_{trace_id}"
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                            client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                            side=Side.SELL.value, qty=base_qty, price=None, status="CREATED",
                            reason_code=ReasonCode.EMERGENCY_EXIT, reason="Emergency exit requested", payload={}
                        )
                        res = ex.place_market_order(symbol=settings.symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                            client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                            event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
                            side=Side.SELL.value, qty=base_qty, price=res.avg_price, status=res.status,
                            reason_code=ReasonCode.EMERGENCY_EXIT, reason="Emergency exit executed", payload=res.raw or {}
                        )
                        save_position(db, settings.symbol, 0.0, None, {"trace_id": trace_id, "note": "emergency_exit"})
                        telegram.send(f"[EMERGENCY_EXIT] trace_id={trace_id} qty={base_qty} symbol={settings.symbol}")
                    set_flag(db, "EMERGENCY_EXIT", "false")
                    continue

                # Hard stop loss
                if base_qty > 0 and avg_entry is not None:
                    stop_price = avg_entry * (1 - settings.hard_stop_loss_pct)
                    if last_price <= stop_price:
                        client_order_id = f"sl_{trace_id}"
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                            client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                            side=Side.SELL.value, qty=base_qty, price=None, status="CREATED",
                            reason_code=ReasonCode.STOP_LOSS,
                            reason=f"Hard stop loss: last={last_price} <= stop={stop_price}",
                            payload={"last_price": last_price, "stop_price": stop_price}
                        )
                        res = ex.place_market_order(symbol=settings.symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                            client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                            event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
                            side=Side.SELL.value, qty=base_qty, price=res.avg_price, status=res.status,
                            reason_code=ReasonCode.STOP_LOSS, reason="Stop loss executed", payload=res.raw or {}
                        )
                        save_position(db, settings.symbol, 0.0, None, {"trace_id": trace_id, "note": "stop_loss"})
                        telegram.send(f"[STOP_LOSS] trace_id={trace_id} qty={base_qty} last={last_price} stop={stop_price}")
                        metrics.orders_total.labels(SERVICE, settings.exchange, settings.symbol, "STOP_LOSS").inc()
                        metrics.last_tick_success.labels(SERVICE, settings.symbol).set(1)
                        continue

                sig = setup_b_signal(latest)
                if sig == "BUY" and base_qty <= 0:
                    qty = 0.001  # MVP fixed sizing; replace with risk model later
                    client_order_id = f"buy_{trace_id}"
                    append_order_event(
                        db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                        client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                        side=Side.BUY.value, qty=qty, price=None, status="CREATED",
                        reason_code=ReasonCode.STRATEGY_SIGNAL, reason="Setup B BUY", payload={"latest": latest}
                    )
                    res = ex.place_market_order(symbol=settings.symbol, side="BUY", qty=qty, client_order_id=client_order_id)
                    append_order_event(
                        db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                        client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                        event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
                        side=Side.BUY.value, qty=qty, price=res.avg_price, status=res.status,
                        reason_code=ReasonCode.STRATEGY_SIGNAL, reason="Order placed", payload=res.raw or {}
                    )
                    entry_price = res.avg_price if res.avg_price is not None else last_price
                    save_position(db, settings.symbol, qty, float(entry_price), {"trace_id": trace_id, "note": "entered"})
                    telegram.send(f"[BUY] trace_id={trace_id} qty={qty} px={entry_price} symbol={settings.symbol}")
                    metrics.orders_total.labels(SERVICE, settings.exchange, settings.symbol, "BUY").inc()
                elif sig == "SELL" and base_qty > 0:
                    qty = base_qty
                    client_order_id = f"sell_{trace_id}"
                    append_order_event(
                        db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                        client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                        side=Side.SELL.value, qty=qty, price=None, status="CREATED",
                        reason_code=ReasonCode.STRATEGY_SIGNAL, reason="Setup B SELL", payload={"latest": latest}
                    )
                    res = ex.place_market_order(symbol=settings.symbol, side="SELL", qty=qty, client_order_id=client_order_id)
                    append_order_event(
                        db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=settings.symbol,
                        client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                        event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
                        side=Side.SELL.value, qty=qty, price=res.avg_price, status=res.status,
                        reason_code=ReasonCode.STRATEGY_SIGNAL, reason="Order placed", payload=res.raw or {}
                    )
                    save_position(db, settings.symbol, 0.0, None, {"trace_id": trace_id, "note": "exited"})
                    telegram.send(f"[SELL] trace_id={trace_id} qty={qty} symbol={settings.symbol}")
                    metrics.orders_total.labels(SERVICE, settings.exchange, settings.symbol, "SELL").inc()

                metrics.last_tick_success.labels(SERVICE, settings.symbol).set(1)
            except Exception as e:
                metrics.last_tick_success.labels(SERVICE, settings.symbol).set(0)
                telegram.send(f"[{SERVICE}] ERROR trace_id={trace_id} err={e}")

if __name__ == "__main__":
    main()
