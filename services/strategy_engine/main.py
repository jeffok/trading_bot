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
from shared.models.order_repo import OrderRepo
from shared.models.position_repo import PositionRepo
from shared.models.service_status_repo import ServiceStatusRepo
from shared.models.system_config_repo import SystemConfigRepo
from shared.strategy.signals import compute_signal_setup_b

SERVICE = "strategy-engine"
VERSION = "0.1.0"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))


def tg_alert(
    telegram: Telegram,
    *,
    level: str,
    event: str,
    title: str,
    trace_id: str,
    exchange: str,
    symbol: str,
    summary_extra: dict,
    payload_extra: dict,
) -> None:
    """只影响 Telegram 展示：中文摘要 + 中文 JSON（交易对 symbol 不翻译）"""

    # 摘要（给人看）
    summary_kv = {
        "level": level,
        "event": event,
        "service": "策略引擎",
        "trace_id": trace_id,
        "exchange": exchange,  # 保持默认值
        "symbol": symbol,      # 保持默认值
        **(summary_extra or {}),
    }

    # JSON（仍然从英文结构构建，但 Telegram 展示会被翻译成中文）
    payload = {
        "level": level,
        "event": event,
        "service": "strategy-engine",
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
    order_repo = OrderRepo(db)
    pos_repo = PositionRepo(db)
    cfg_repo = SystemConfigRepo(db)
    svc_repo = ServiceStatusRepo(db)

    ex = make_exchange(settings)

    logger.info("Start %s version=%s exchange=%s symbol=%s", SERVICE, VERSION, settings.exchange, settings.symbol)

    while True:
        trace_id = new_trace_id("tick")
        try:
            with distributed_lock(r, key=f"lock:{SERVICE}:{settings.symbol}", ttl_seconds=25):
                svc_repo.heartbeat(service_name=SERVICE, instance_id=settings.instance_id)

                halt = cfg_repo.get_bool("HALT_TRADING", default=False)
                if halt:
                    logger.warning("HALT_TRADING=true, skip tick trace_id=%s", trace_id)
                    tg_alert(
                        telegram,
                        level="INFO",
                        event="HALT_SKIP_TICK",
                        title=f"⏸️ 已暂停：跳过本轮交易 {settings.symbol}",
                        trace_id=trace_id,
                        exchange=settings.exchange,
                        symbol=settings.symbol,
                        summary_extra={"说明": "HALT_TRADING=true"},
                        payload_extra={"reason_code": "ADMIN_HALT"},
                    )
                    time.sleep(settings.strategy_tick_seconds)
                    continue

                latest = cache_repo.get_latest(symbol=settings.symbol, interval_minutes=settings.interval_minutes)
                if not latest:
                    logger.warning("No market data cache yet, skip tick trace_id=%s", trace_id)
                    tg_alert(
                        telegram,
                        level="WARN",
                        event="NO_MARKET_DATA",
                        title=f"⚠️ 无行情数据：跳过本轮 {settings.symbol}",
                        trace_id=trace_id,
                        exchange=settings.exchange,
                        symbol=settings.symbol,
                        summary_extra={"说明": "market_data_cache 暂无数据"},
                        payload_extra={"reason_code": "DATA_SYNC"},
                    )
                    time.sleep(settings.strategy_tick_seconds)
                    continue

                cfg = {
                    "rsi_buy": cfg_repo.get_float("RSI_BUY", default=settings.rsi_buy),
                    "rsi_sell": cfg_repo.get_float("RSI_SELL", default=settings.rsi_sell),
                    "stop_loss_pct": cfg_repo.get_float("STOP_LOSS_PCT", default=settings.stop_loss_pct),
                    "qty": cfg_repo.get_float("TRADE_QTY", default=settings.trade_qty),
                }

                base_qty, avg_entry = pos_repo.get_latest(symbol=settings.symbol)

                # 紧急退出
                if cfg_repo.get_bool("EMERGENCY_EXIT", default=False):
                    if base_qty > 0:
                        client_order_id = f"emergency_exit_{trace_id}"
                        res = ex.place_market_order(symbol=settings.symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)

                        order_repo.append_event(
                            trace_id=trace_id,
                            service=SERVICE,
                            exchange=settings.exchange,
                            symbol=settings.symbol,
                            client_order_id=client_order_id,
                            exchange_order_id=res.exchange_order_id,
                            event_type="CREATED",
                            side="SELL",
                            qty=base_qty,
                            price=None,
                            status="FILLED",
                            reason_code="EMERGENCY_EXIT",
                            reason="Emergency exit executed",
                            payload={"exchange_response": res.raw},
                        )
                        pos_repo.append_snapshot(symbol=settings.symbol, base_qty=0.0, avg_entry_price=0.0, meta={"reason": "EMERGENCY_EXIT"})
                        cfg_repo.set("EMERGENCY_EXIT", "false")

                        tg_alert(
                            telegram,
                            level="CRITICAL",
                            event="EMERGENCY_EXIT_EXECUTED",
                            title=f"🆘 紧急退出已执行 {settings.symbol}",
                            trace_id=trace_id,
                            exchange=settings.exchange,
                            symbol=settings.symbol,
                            summary_extra={"qty": base_qty},
                            payload_extra={
                                "reason_code": "EMERGENCY_EXIT",
                                "side": "SELL",
                                "qty": base_qty,
                                "client_order_id": client_order_id,
                                "exchange_order_id": res.exchange_order_id,
                            },
                        )
                    time.sleep(settings.strategy_tick_seconds)
                    continue

                last_price = float(latest["close_price"])
                stop_price = (avg_entry or 0.0) * (1.0 - float(cfg["stop_loss_pct"]))

                # 止损
                if base_qty > 0 and avg_entry and last_price <= stop_price:
                    client_order_id = f"stoploss_{trace_id}"
                    res = ex.place_market_order(symbol=settings.symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)

                    order_repo.append_event(
                        trace_id=trace_id,
                        service=SERVICE,
                        exchange=settings.exchange,
                        symbol=settings.symbol,
                        client_order_id=client_order_id,
                        exchange_order_id=res.exchange_order_id,
                        event_type="CREATED",
                        side="SELL",
                        qty=base_qty,
                        price=None,
                        status="FILLED",
                        reason_code="STOP_LOSS",
                        reason="Hard stop loss triggered",
                        payload={"exchange_response": res.raw, "last_price": last_price, "stop_price": stop_price},
                    )
                    pos_repo.append_snapshot(symbol=settings.symbol, base_qty=0.0, avg_entry_price=0.0, meta={"reason": "STOP_LOSS"})

                    tg_alert(
                        telegram,
                        level="CRITICAL",
                        event="STOP_LOSS",
                        title=f"🚨 触发止损 {settings.symbol}",
                        trace_id=trace_id,
                        exchange=settings.exchange,
                        symbol=settings.symbol,
                        summary_extra={"last_price": last_price, "stop_price": stop_price, "qty": base_qty},
                        payload_extra={
                            "reason_code": "STOP_LOSS",
                            "side": "SELL",
                            "qty": base_qty,
                            "last_price": last_price,
                            "stop_price": stop_price,
                            "avg_entry_price": avg_entry,
                            "client_order_id": client_order_id,
                            "exchange_order_id": res.exchange_order_id,
                        },
                    )
                    time.sleep(settings.strategy_tick_seconds)
                    continue

                signal = compute_signal_setup_b(
                    close_price=last_price,
                    ema_fast=float(latest["ema_fast"]),
                    ema_slow=float(latest["ema_slow"]),
                    rsi=float(latest["rsi"]),
                    base_qty=float(base_qty),
                    rsi_buy=float(cfg["rsi_buy"]),
                    rsi_sell=float(cfg["rsi_sell"]),
                )

                # 开仓
                if signal == "BUY":
                    qty = float(cfg["qty"])
                    if qty > 0:
                        client_order_id = f"buy_{trace_id}"
                        res = ex.place_market_order(symbol=settings.symbol, side="BUY", qty=qty, client_order_id=client_order_id)
                        entry_price = float(res.fill_price or last_price)

                        order_repo.append_event(
                            trace_id=trace_id,
                            service=SERVICE,
                            exchange=settings.exchange,
                            symbol=settings.symbol,
                            client_order_id=client_order_id,
                            exchange_order_id=res.exchange_order_id,
                            event_type="CREATED",
                            side="BUY",
                            qty=qty,
                            price=None,
                            status="FILLED",
                            reason_code="STRATEGY_SIGNAL",
                            reason="Setup B BUY",
                            payload={"exchange_response": res.raw, "signal": "BUY"},
                        )
                        pos_repo.append_snapshot(symbol=settings.symbol, base_qty=qty, avg_entry_price=entry_price, meta={"signal": "BUY"})

                        tg_alert(
                            telegram,
                            level="INFO",
                            event="BUY_FILLED",
                            title=f"✅ 开仓成交 {settings.symbol}",
                            trace_id=trace_id,
                            exchange=settings.exchange,
                            symbol=settings.symbol,
                            summary_extra={"qty": qty, "entry_price": entry_price},
                            payload_extra={
                                "reason_code": "STRATEGY_SIGNAL",
                                "side": "BUY",
                                "qty": qty,
                                "entry_price": entry_price,
                                "client_order_id": client_order_id,
                                "exchange_order_id": res.exchange_order_id,
                            },
                        )

                # 平仓
                elif signal == "SELL":
                    if base_qty > 0:
                        qty = float(base_qty)
                        client_order_id = f"sell_{trace_id}"
                        res = ex.place_market_order(symbol=settings.symbol, side="SELL", qty=qty, client_order_id=client_order_id)

                        order_repo.append_event(
                            trace_id=trace_id,
                            service=SERVICE,
                            exchange=settings.exchange,
                            symbol=settings.symbol,
                            client_order_id=client_order_id,
                            exchange_order_id=res.exchange_order_id,
                            event_type="CREATED",
                            side="SELL",
                            qty=qty,
                            price=None,
                            status="FILLED",
                            reason_code="STRATEGY_SIGNAL",
                            reason="Setup B SELL",
                            payload={"exchange_response": res.raw, "signal": "SELL"},
                        )
                        pos_repo.append_snapshot(symbol=settings.symbol, base_qty=0.0, avg_entry_price=0.0, meta={"signal": "SELL"})

                        tg_alert(
                            telegram,
                            level="INFO",
                            event="SELL_FILLED",
                            title=f"✅ 平仓成交 {settings.symbol}",
                            trace_id=trace_id,
                            exchange=settings.exchange,
                            symbol=settings.symbol,
                            summary_extra={"qty": qty},
                            payload_extra={
                                "reason_code": "STRATEGY_SIGNAL",
                                "side": "SELL",
                                "qty": qty,
                                "client_order_id": client_order_id,
                                "exchange_order_id": res.exchange_order_id,
                            },
                        )

                time.sleep(settings.strategy_tick_seconds)

        except Exception as e:
            logger.exception("Unhandled exception in tick loop trace_id=%s", trace_id)
            tg_alert(
                telegram,
                level="ERROR",
                event="UNHANDLED_EXCEPTION",
                title=f"❌ 策略异常 {settings.symbol}",
                trace_id=trace_id,
                exchange=settings.exchange,
                symbol=settings.symbol,
                summary_extra={"错误": str(e)[:200]},
                payload_extra={"reason_code": "SYSTEM", "error": str(e)},
            )
            time.sleep(2.0)


if __name__ == "__main__":
    main()
