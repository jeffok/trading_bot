
from __future__ import annotations

import datetime
import json
import os
import time
import threading
from pathlib import Path
from typing import Optional

from shared.config import Settings, load_settings
from shared.db import MariaDB, migrate
from shared.exchange import make_exchange
from shared.exchange.base import ExchangeClient
from shared.exchange.errors import RateLimitError
from shared.ai import OnlineLogisticRegression, SGDClassifierCompat, load_current_model_blob, save_current_model_blob
from shared.domain.system_config import get_system_config, write_system_config
from shared.domain.control_commands import fetch_new_control_commands, mark_control_command_processed
from shared.logging import get_logger, new_trace_id
from shared.redis import distributed_lock, redis_client, LeaderElector
from shared.telemetry import Metrics, Telegram, start_metrics_http_server, log_action, build_system_summary, send_system_alert
from shared.telemetry.trade_alerts import build_trade_summary, send_trade_alert
from shared.domain.enums import OrderEventType, ReasonCode, Side
from shared.domain.heartbeat import upsert_service_status
from shared.domain.instance import get_instance_id
from shared.domain.runtime_config import RuntimeConfig
from shared.domain.events import append_order_event, get_first_event_created_at
from shared.domain.idempotency import make_client_order_id
from shared.domain.time import next_tick_sleep_seconds, HK

SERVICE = "strategy-engine"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))

def _normalize_status(s: str) -> str:
    return str(s or "").upper()


def _event_type_from_status(status: str) -> OrderEventType:
    """将交易所返回的 status 映射为统一事件类型（用于 SUBMITTED / PARTIAL / 终态）。"""
    s = _normalize_status(status)
    if s in ("FILLED", "CLOSED"):
        return OrderEventType.FILLED
    if s in ("CANCELED", "CANCELLED", "CANCELED_BY_USER"):
        return OrderEventType.CANCELED
    if s in ("REJECTED",):
        return OrderEventType.REJECTED
    if s in ("PARTIALLY_FILLED", "PARTIAL", "PARTIALLYFILLED"):
        return OrderEventType.PARTIAL
    if s in ("ERROR", "FAILED", "EXPIRED"):
        return OrderEventType.ERROR
    return OrderEventType.SUBMITTED



def reconcile_stale_orders(db: MariaDB, ex, *, exchange_name: str, max_age_seconds: int, metrics: Metrics, telegram: Telegram) -> int:
    """Best-effort reconciliation: query exchange for stale orders.

    It scans orders whose latest event is CREATED/SUBMITTED and older than max_age_seconds, then:
    - writes a terminal event if detected (FILLED/CANCELED/ERROR)
    - writes a RECONCILED event with the observed status
    """
    metrics.reconcile_runs_total.labels(SERVICE).inc()

    rows = db.fetch_all(
        """
        SELECT oe.*
        FROM order_events oe
        JOIN (
          SELECT exchange, symbol, client_order_id, MAX(id) AS max_id
          FROM order_events
          GROUP BY exchange, symbol, client_order_id
        ) t ON oe.id = t.max_id
        WHERE oe.event_type IN ('CREATED','SUBMITTED','PARTIAL','RECONCILED')
          AND oe.created_at < (UTC_TIMESTAMP() - INTERVAL %s SECOND)
        LIMIT 200
        """,
        (int(max_age_seconds),),
    )

    fixed = 0
    for r in rows or []:
        symbol = str(r["symbol"])
        client_order_id = str(r["client_order_id"])
        exchange_order_id = r.get("exchange_order_id")
        try:
            st = ex.get_order_status(symbol=symbol, client_order_id=client_order_id, exchange_order_id=exchange_order_id)
            status_u = _normalize_status(getattr(st, "status", ""))

            metrics.reconcile_orders_total.labels(SERVICE, symbol).inc()

            terminal_event = None
            partial_event = False

            if status_u in ("FILLED", "CLOSED"):
                terminal_event = OrderEventType.FILLED
            elif status_u in ("CANCELED", "CANCELLED", "CANCELED_BY_USER"):
                terminal_event = OrderEventType.CANCELED
            elif status_u in ("REJECTED",):
                terminal_event = OrderEventType.REJECTED
            elif status_u in ("PARTIALLY_FILLED", "PARTIAL", "PARTIALLYFILLED"):
                partial_event = True
            elif status_u in ("EXPIRED", "ERROR", "FAILED"):
                terminal_event = OrderEventType.ERROR
            payload = getattr(st, "raw", None) or {}
            qty = float(getattr(st, "filled_qty", 0.0) or 0.0)
            avg_price = getattr(st, "avg_price", None)

            trace_id = new_trace_id("reconcile")

            # PARTIAL：如果交易所返回“部分成交”，写入一次 PARTIAL 事件（幂等）。
            if partial_event and qty > 0:
                append_order_event(
                    db,
                    trace_id=trace_id,
                    service=SERVICE,
                    exchange=exchange_name,
                    symbol=symbol,
                    client_order_id=client_order_id,
                    exchange_order_id=str(getattr(st, "exchange_order_id", "") or exchange_order_id),
                    event_type=OrderEventType.PARTIAL,
                    side=str(r.get("side") or ""),
                    qty=qty,
                    price=float(avg_price) if avg_price is not None else None,
                    status=status_u,
                    reason_code=ReasonCode.RECONCILE,
                    reason="Reconciled partial fill",
                    payload=payload,
                )


            if terminal_event is not None:
                inserted = append_order_event(
                    db,
                    trace_id=trace_id,
                    service=SERVICE,
                    exchange=exchange_name,
                    symbol=symbol,
                    client_order_id=client_order_id,
                    exchange_order_id=str(getattr(st, "exchange_order_id", "") or exchange_order_id),
                    event_type=terminal_event,
                    side=str(r.get("side") or ""),
                    qty=qty or float(r.get("qty") or 0.0),
                    price=float(avg_price) if avg_price is not None else None,
                    status=status_u,
                    reason_code=ReasonCode.RECONCILE,
                    reason="Reconciled terminal status",
                    payload=payload,
                )
                if inserted:
                    # e2e latency：从 CREATED 到首次进入终态事件
                    created_at = get_first_event_created_at(
                        db, exchange=exchange_name, symbol=symbol, client_order_id=client_order_id
                    )
                    if created_at is not None:
                        try:
                            now = datetime.datetime.utcnow()
                            latency = max(0.0, (now - created_at).total_seconds())
                            metrics.order_e2e_latency_seconds.labels(
                                SERVICE, exchange_name, symbol, terminal_event.value
                            ).observe(latency)
                        except Exception:
                            pass
                    metrics.reconcile_fixed_total.labels(SERVICE, symbol, status_u).inc()
                    fixed += 1

            append_order_event(
                db,
                trace_id=trace_id,
                service=SERVICE,
                exchange=exchange_name,
                symbol=symbol,
                client_order_id=client_order_id,
                exchange_order_id=str(getattr(st, "exchange_order_id", "") or exchange_order_id),
                event_type=OrderEventType.RECONCILED,
                side=str(r.get("side") or ""),
                qty=float(r.get("qty") or 0.0),
                price=None,
                status=status_u,
                reason_code=ReasonCode.RECONCILE,
                reason="Reconciled order status",
                payload=payload,
            )
        except Exception as e:
            try:
                send_system_alert(
                    telegram,
                    title="❌ Reconcile 异常",
                    summary_kv=build_system_summary(
                        event="RECONCILE_ERROR",
                        trace_id=trace_id,
                        exchange=None,
                        level="ERROR",
                        extra={"symbol": symbol, "client_order_id": client_order_id, "error": str(e)[:200]},
                    ),
                    payload={"error": str(e)},
                )
                log_action(logger, action="RECONCILE_ERROR", trace_id=trace_id, reason_code="ERROR", reason=str(e)[:200], client_order_id=client_order_id, extra={"symbol": symbol})
            except Exception:
                pass
            continue

    return fixed

def apply_control_commands(db: MariaDB, telegram: Telegram, *, exchange: str | None, trace_id: str) -> None:
    """Consume NEW control_commands and apply to system_config (minimal)."""
    cmds = fetch_new_control_commands(db, limit=50)
    for c in cmds:
        cid = int(c.get("id") or 0)
        command = str(c.get("command") or "").strip().upper()
        payload = c.get("payload") or {}
        try:
            if command in ("HALT", "HALT_TRADING"):
                set_flag(db, "HALT_TRADING", "true")
                send_system_alert(telegram, title="系统熔断/暂停", summary_kv=build_system_summary(event="HALT", trace_id=trace_id, exchange=exchange, actor=str(payload.get("actor") or "" ) or None, reason_code=payload.get("reason_code"), reason=payload.get("reason"), extra={"source": "control_commands"}), payload={"payload": payload})
            elif command in ("RESUME", "RESUME_TRADING"):
                set_flag(db, "HALT_TRADING", "false")
                send_system_alert(telegram, title="系统恢复交易", summary_kv=build_system_summary(event="RESUME", trace_id=trace_id, exchange=exchange, actor=str(payload.get("actor") or "" ) or None, reason_code=payload.get("reason_code"), reason=payload.get("reason"), extra={"source": "control_commands"}), payload={"payload": payload})
            elif command in ("EMERGENCY_EXIT", "EMERGENCY"):
                set_flag(db, "EMERGENCY_EXIT", "true")
                send_system_alert(
                    telegram,
                    title="紧急平仓触发",
                    summary_kv=build_system_summary(
                        event="EMERGENCY_EXIT",
                        trace_id=trace_id,
                        exchange=exchange,
                        actor=str(payload.get("actor") or "") or None,
                        reason_code=payload.get("reason_code"),
                        reason=payload.get("reason"),
                        extra={"source": "control_commands"},
                    ),
                    payload={"payload": payload},
                )
                log_action(
                    logger,
                    action="EMERGENCY_EXIT_TRIGGERED",
                    trace_id=trace_id,
                    reason_code=payload.get("reason_code") or "CONTROL_COMMAND",
                    reason=str(payload.get("reason") or "emergency exit triggered"),
                    client_order_id=None,
                    extra={"source": "control_commands"},
                )
            elif command in ("UPDATE_CONFIG", "SET_CONFIG"):
                # payload: {key,value}
                k = str(payload.get("key") or "").strip()
                v = str(payload.get("value") or "").strip()
                if k:
                    write_system_config(db, actor=str(payload.get("actor") or "control"), key=k, value=v, trace_id=trace_id,
                                       reason_code=str(payload.get("reason_code") or ReasonCode.SYSTEM.value),
                                       reason=str(payload.get("reason") or "control_command"))
            # mark processed
            if cid > 0:
                mark_control_command_processed(db, command_id=cid, status="PROCESSED")
        except Exception:
            if cid > 0:
                mark_control_command_processed(db, command_id=cid, status="ERROR")

def get_flag(db: MariaDB, key: str, default: str = "false") -> str:
    row = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
    return (row["value"] if row else default).strip().lower()

def set_flag(db: MariaDB, key: str, value: str) -> None:
    db.execute("INSERT INTO system_config(`key`,`value`) VALUES (%s,%s) ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)", (key, value))

def latest_cache(db: MariaDB, symbol: str, interval_minutes: int, feature_version: int = 1):
    return db.fetch_one(
        """
        SELECT m.open_time_ms, m.close_price, c.ema_fast, c.ema_slow, c.rsi, c.features_json
        FROM market_data m
        LEFT JOIN market_data_cache c
          ON c.symbol=m.symbol AND c.interval_minutes=m.interval_minutes AND c.open_time_ms=m.open_time_ms AND c.feature_version=%s
        WHERE m.symbol=%s AND m.interval_minutes=%s
        ORDER BY m.open_time_ms DESC
        LIMIT 1
        """,
        (symbol, interval_minutes, int(feature_version or 1)),
    )

def last_two_cache(db: MariaDB, symbol: str, interval_minutes: int, feature_version: int = 1):
    """Return (latest, prev) cache rows. prev may be None."""
    rows = db.fetch_all(
        """
        SELECT m.open_time_ms, m.close_price, c.ema_fast, c.ema_slow, c.rsi, c.features_json
        FROM market_data m
        LEFT JOIN market_data_cache c
          ON c.symbol=m.symbol AND c.interval_minutes=m.interval_minutes AND c.open_time_ms=m.open_time_ms AND c.feature_version=%s
        WHERE m.symbol=%s AND m.interval_minutes=%s
        ORDER BY m.open_time_ms DESC
        LIMIT 2
        """,
        (symbol, interval_minutes, int(feature_version or 1)),
    ) or []
    latest = rows[0] if len(rows) >= 1 else None
    prev = rows[1] if len(rows) >= 2 else None
    return latest, prev

def get_position(db: MariaDB, symbol: str):
    return db.fetch_one(
        """
        SELECT id, created_at, base_qty, avg_entry_price, meta_json
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

def _stop_client_order_id(base_open_client_order_id: str, seq: int = 1) -> str:
    # seq=1 -> _SL, seq=2 -> _SL2, ...
    if seq <= 1:
        return f"{base_open_client_order_id}_SL"
    return f"{base_open_client_order_id}_SL{int(seq)}"


def _append_stop_event(
    db: MariaDB,
    *,
    trace_id: str,
    exchange_name: str,
    symbol: str,
    client_order_id: str,
    exchange_order_id: str | None,
    event_type: OrderEventType,
    qty: float,
    stop_price: float | None,
    status: str,
    reason_code: ReasonCode,
    reason: str,
    payload: dict,
) -> None:
    # 注意：stop 订单 side 固定为 SELL（用于平多仓）
    append_order_event(
        db,
        trace_id=trace_id,
        service=SERVICE,
        exchange=exchange_name,
        symbol=symbol,
        client_order_id=client_order_id,
        exchange_order_id=exchange_order_id,
        event_type=event_type,
        side=Side.SELL.value,
        qty=float(qty),
        price=float(stop_price) if stop_price is not None else None,
        status=status,
        reason_code=reason_code,
        reason=reason,
        payload=payload,
    )


def _arm_protective_stop_with_retry(
    *,
    exchange: ExchangeClient,
    db: MariaDB,
    metrics: Metrics,
    telegram: Telegram,
    settings: Settings,
    runtime_cfg: RuntimeConfig,
    symbol: str,
    qty: float,
    stop_price: float,
    trace_id: str,
    trade_id: int,
    base_open_client_order_id: str,
    action: str = "ARM",
    seq: int = 1,
) -> tuple[str | None, str | None]:
    """挂保护止损单（STOP_MARKET）：重试 + 失败降级（回退为内部硬止损）。

    action: ARM | REARM（用于指标与 reason_code）
    seq: client_order_id 的序号（避免与历史冲突）
    """
    if settings.exchange == "paper":
        return (None, None)

    stop_client_order_id = _stop_client_order_id(base_open_client_order_id, seq=seq)
    max_retries = max(1, int(runtime_cfg.stop_arm_max_retries))
    base_backoff = float(runtime_cfg.stop_arm_backoff_base_seconds)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            res = exchange.place_stop_market_order(
                symbol=symbol,
                side=Side.SELL.value,
                qty=float(qty),
                stop_price=float(stop_price),
                client_order_id=stop_client_order_id,
                reduce_only=True,
            )
            stop_exchange_order_id = res.exchange_order_id

            # 审计：记录为 CREATED（或 SUBMITTED），与普通下单一致
            reason_code = ReasonCode.STOP_LOSS_ARMED if action == "ARM" else ReasonCode.STOP_LOSS_REARMED
            _append_stop_event(
                db,
                trace_id=trace_id,
                exchange_name=exchange.name,
                symbol=symbol,
                client_order_id=stop_client_order_id,
                exchange_order_id=stop_exchange_order_id,
                event_type=OrderEventType.CREATED,
                qty=float(qty),
                stop_price=float(stop_price),
                status=str(res.status or "CREATED"),
                reason_code=reason_code,
                reason="Protective stop-market order armed" if action == "ARM" else "Protective stop-market order re-armed",
                payload={"stop_price": float(stop_price), "reduce_only": True, "order_type": "STOP_MARKET", "attempt": attempt, "seq": seq},
            )

            _update_trade_stop_order(
                db,
                trade_id=trade_id,
                stop_client_order_id=stop_client_order_id,
                stop_exchange_order_id=stop_exchange_order_id,
                stop_order_type="STOP_MARKET",
            )

            try:
                metrics.stop_armed_total.labels(SERVICE, exchange.name, symbol, action).inc()
            except Exception:
                pass

            return (stop_client_order_id, stop_exchange_order_id)

        except RateLimitError as e:
            last_err = e
            try:
                metrics.stop_arm_failed_total.labels(SERVICE, exchange.name, symbol, action).inc()
            except Exception:
                pass
            _append_stop_event(
                db,
                trace_id=trace_id,
                exchange_name=exchange.name,
                symbol=symbol,
                client_order_id=stop_client_order_id,
                exchange_order_id=None,
                event_type=OrderEventType.ERROR,
                qty=float(qty),
                stop_price=float(stop_price),
                status="RATE_LIMIT",
                reason_code=ReasonCode.RATE_LIMIT,
                reason=f"Rate limited when arming stop (attempt {attempt}/{max_retries})",
                payload={"stop_price": float(stop_price), "attempt": attempt, "max_retries": max_retries, "error": str(e)},
            )
            # backoff
            sleep_s = float(getattr(e, "retry_after_seconds", 0.0) or 0.0)
            if sleep_s <= 0:
                sleep_s = min(5.0, base_backoff * (2 ** (attempt - 1)))
            time.sleep(max(0.1, sleep_s))
            continue
        except Exception as e:
            last_err = e
            try:
                metrics.stop_arm_failed_total.labels(SERVICE, exchange.name, symbol, action).inc()
            except Exception:
                pass
            _append_stop_event(
                db,
                trace_id=trace_id,
                exchange_name=exchange.name,
                symbol=symbol,
                client_order_id=stop_client_order_id,
                exchange_order_id=None,
                event_type=OrderEventType.ERROR,
                qty=float(qty),
                stop_price=float(stop_price),
                status="ERROR",
                reason_code=ReasonCode.STOP_LOSS_ARM_FAILED if action == "ARM" else ReasonCode.STOP_LOSS_REARM_FAILED,
                reason=f"Failed to arm protective stop (attempt {attempt}/{max_retries}): {e}",
                payload={"stop_price": float(stop_price), "attempt": attempt, "max_retries": max_retries},
            )
            time.sleep(max(0.05, min(5.0, base_backoff * (2 ** (attempt - 1)))))
            continue

    # all retries failed -> downgrade
    summary_kv = build_trade_summary(
        event="STOP_ARM_FAILED_FALLBACK",
        trace_id=trace_id,
        exchange=exchange.name,
        symbol=symbol,
        qty=qty,
        stop_price=stop_price,
        reason="Protective stop placement failed; fallback to polling",
        extra={"max_retries": max_retries},
    )
    send_trade_alert(
        telegram,
        title="保护止损单挂单失败（已降级）",
        summary_kv=summary_kv,
        payload={"error": str(last_err) if last_err else "unknown"},
    )
    log_action(
        logger,
        "STOP_ARM_FAILED_FALLBACK",
        trace_id=trace_id,
        exchange=exchange.name,
        symbol=symbol,
        qty=qty,
        stop_price=stop_price,
        error=str(last_err) if last_err else "unknown",
    )
    return (None, None)


def _cancel_protective_stop(
    *,
    exchange: ExchangeClient,
    db: MariaDB,
    symbol: str,
    trace_id: str,
    meta: dict,
    reason_code: ReasonCode,
    reason: str,
) -> None:
    """在主动平仓/紧急退出前取消保护止损单，避免“平仓后止损再触发”导致反向开仓。"""
    stop_client_order_id = meta.get("stop_client_order_id")
    stop_exchange_order_id = meta.get("stop_exchange_order_id")
    trade_id = int(meta.get("trade_id") or 0)
    if not stop_client_order_id:
        return
    try:
        ok = exchange.cancel_order(symbol=symbol, client_order_id=str(stop_client_order_id), exchange_order_id=stop_exchange_order_id)
        _append_stop_event(
            db,
            trace_id=trace_id,
            exchange_name=exchange.name,
            symbol=symbol,
            client_order_id=str(stop_client_order_id),
            exchange_order_id=str(stop_exchange_order_id) if stop_exchange_order_id else None,
            event_type=OrderEventType.CANCELED if ok else OrderEventType.ERROR,
            qty=float(meta.get("base_qty") or 0.0),
            stop_price=float(meta.get("stop_price")) if meta.get("stop_price") is not None else None,
            status="CANCELED" if ok else "ERROR",
            reason_code=reason_code,
            reason=reason if ok else f"{reason} (cancel failed)",
            payload={"ok": bool(ok)},
        )
    except Exception as e:
        _append_stop_event(
            db,
            trace_id=trace_id,
            exchange_name=exchange.name,
            symbol=symbol,
            client_order_id=str(stop_client_order_id),
            exchange_order_id=str(stop_exchange_order_id) if stop_exchange_order_id else None,
            event_type=OrderEventType.ERROR,
            qty=float(meta.get("base_qty") or 0.0),
            stop_price=float(meta.get("stop_price")) if meta.get("stop_price") is not None else None,
            status="ERROR",
            reason_code=ReasonCode.SYSTEM,
            reason=f"{reason} (exception): {e}",
            payload={"error": str(e)},
        )

    # clear in trade_logs (best-effort)
    if trade_id > 0:
        _update_trade_stop_order(db, trade_id=trade_id, stop_client_order_id=None, stop_exchange_order_id=None, stop_order_type=None)
    meta["stop_client_order_id"] = None
    meta["stop_exchange_order_id"] = None


def _ensure_protective_stop(
    *,
    exchange: ExchangeClient,
    db: MariaDB,
    metrics: Metrics,
    telegram: Telegram,
    settings: Settings,
    runtime_cfg: RuntimeConfig,
    symbol: str,
    base_qty: float,
    avg_entry: float | None,
    pos_row: dict,
    trace_id: str,
) -> tuple[bool, dict]:
    """确保保护止损单处于“有效状态”。

    返回 (position_closed, new_meta)
    - position_closed=True 表示交易所止损单已成交，本函数已写入止损订单事件；调用方仍需关闭 trade_logs 并将 position 置 0
    """
    meta = _parse_json_maybe(pos_row.get("meta_json") if pos_row else None)
    meta["base_qty"] = float(base_qty)
    now_ms = int(time.time() * 1000)

    trade_id = _find_open_trade_id(db, symbol, meta)
    if trade_id > 0:
        meta["trade_id"] = int(trade_id)

        # Recovery: if position meta lost stop order ids, try to restore from trade_logs
        if not meta.get("stop_client_order_id"):
            try:
                trade_stop = _fetch_trade_stop_order(db, int(trade_id))
                if trade_stop.get("stop_client_order_id"):
                    meta["stop_client_order_id"] = trade_stop.get("stop_client_order_id")
                    meta["stop_exchange_order_id"] = trade_stop.get("stop_exchange_order_id")
            except Exception:
                pass

    # compute stop_price if missing
    if meta.get("stop_price") is None:
        try:
            stop_dist_pct = float(meta.get("stop_dist_pct") or settings.hard_stop_loss_pct)
        except Exception:
            stop_dist_pct = float(settings.hard_stop_loss_pct)
        meta["stop_dist_pct"] = float(stop_dist_pct)
        if avg_entry is not None:
            meta["stop_price"] = float(avg_entry) * (1.0 - float(stop_dist_pct))

    stop_price = meta.get("stop_price")
    if stop_price is None:
        # can't arm without stop_price
        return (False, meta)

    # 1) if stop already exists -> poll status
    stop_client_order_id = meta.get("stop_client_order_id")
    stop_exchange_order_id = meta.get("stop_exchange_order_id")
    if stop_client_order_id:
        try:
            st = exchange.get_order_status(symbol=symbol, client_order_id=str(stop_client_order_id), exchange_order_id=stop_exchange_order_id)
            status_u = _normalize_status(getattr(st, "status", ""))

            if status_u in ("FILLED", "CLOSED"):
                # terminal: stop filled
                # terminal: stop filled
                _append_stop_event(
                    db,
                    trace_id=trace_id,
                    exchange_name=exchange.name,
                    symbol=symbol,
                    client_order_id=str(stop_client_order_id),
                    exchange_order_id=str(getattr(st, "exchange_order_id", "") or stop_exchange_order_id) if (getattr(st, "exchange_order_id", None) or stop_exchange_order_id) else None,
                    event_type=OrderEventType.FILLED,
                    qty=float(getattr(st, "filled_qty", 0.0) or base_qty or 0.0),
                    stop_price=float(stop_price),
                    status=status_u,
                    reason_code=ReasonCode.STOP_LOSS,
                    reason="Protective stop order filled",
                    payload=getattr(st, "raw", None) or {},
                )
                meta["_stop_fill"] = {
                    "avg_price": getattr(st, "avg_price", None),
                    "pnl_usdt": getattr(st, "pnl_usdt", None),
                    "fee_usdt": getattr(st, "fee_usdt", None),
                    "filled_qty": getattr(st, "filled_qty", None),
                    "raw": getattr(st, "raw", None) or {},
                }
                return (True, meta)

            if status_u in ("REJECTED", "EXPIRED", "ERROR", "FAILED", "CANCELED", "CANCELLED", "CANCELED_BY_USER"):
                try:
                    metrics.stop_order_invalid_total.labels(SERVICE, exchange.name, symbol, status_u).inc()
                except Exception:
                    pass

                # stop became invalid -> clear and try re-arm with cooldown/limit
                _append_stop_event(
                    db,
                    trace_id=trace_id,
                    exchange_name=exchange.name,
                    symbol=symbol,
                    client_order_id=str(stop_client_order_id),
                    exchange_order_id=str(getattr(st, "exchange_order_id", "") or stop_exchange_order_id) if (getattr(st, "exchange_order_id", None) or stop_exchange_order_id) else None,
                    event_type=OrderEventType.ERROR,
                    qty=float(base_qty),
                    stop_price=float(stop_price),
                    status=status_u,
                    reason_code=ReasonCode.STOP_ORDER_INVALID,
                    reason=f"Protective stop became invalid: {status_u}",
                    payload=getattr(st, "raw", None) or {},
                )

                # clear
                meta["stop_client_order_id"] = None
                meta["stop_exchange_order_id"] = None
                if trade_id > 0:
                    _update_trade_stop_order(db, trade_id=trade_id, stop_client_order_id=None, stop_exchange_order_id=None, stop_order_type=None)

                # rearm gating
                rearm_attempts = int(meta.get("stop_rearm_attempts") or 0)
                last_rearm_ms = int(meta.get("stop_rearm_last_ms") or 0)
                if rearm_attempts >= int(runtime_cfg.stop_rearm_max_attempts):
                    meta["stop_arming_disabled"] = True
                    try:
                        send_system_alert(
                            telegram,
                            title="⚠️ 停止保护止损重挂（达到上限）",
                            summary_kv=build_system_summary(
                                event="STOP_ARMING_DISABLED",
                                trace_id=trace_id,
                                exchange=exchange,
                                level="WARN",
                                extra={"symbol": symbol, "attempts": rearm_attempts},
                            ),
                            payload={"symbol": symbol, "attempts": rearm_attempts},
                        )
                        log_action(logger, action="STOP_ARMING_DISABLED", trace_id=trace_id, reason_code="WARN", reason="stop rearm max attempts", client_order_id=None, extra={"symbol": symbol, "attempts": rearm_attempts})
                    except Exception:
                        pass
                    return (False, meta)

                if (now_ms - last_rearm_ms) < int(runtime_cfg.stop_rearm_cooldown_seconds) * 1000:
                    return (False, meta)

                base_open = str(meta.get("open_client_order_id") or meta.get("entry_client_order_id") or "")
                if not base_open:
                    # 没有 entry 的 client_order_id，只能降级
                    return (False, meta)

                meta["stop_rearm_attempts"] = rearm_attempts + 1
                meta["stop_rearm_last_ms"] = now_ms

                stop_seq = int(meta.get("stop_rearm_seq") or 1) + 1
                meta["stop_rearm_seq"] = stop_seq

                new_stop_cid, new_stop_eid = _arm_protective_stop_with_retry(
                    exchange=exchange,
                    db=db,
                    metrics=metrics,
                    telegram=telegram,
                    settings=settings,
                    runtime_cfg=runtime_cfg,
                    symbol=symbol,
                    qty=float(base_qty),
                    stop_price=float(stop_price),
                    trace_id=trace_id,
                    trade_id=int(trade_id),
                    base_open_client_order_id=base_open,
                    action="REARM",
                    seq=int(stop_seq),
                )
                if new_stop_cid and new_stop_eid:
                    meta["stop_client_order_id"] = new_stop_cid
                    meta["stop_exchange_order_id"] = new_stop_eid
                else:
                    # downgrade to internal stop
                    meta["stop_arming_disabled"] = True
                return (False, meta)

            # otherwise: still active (NEW/OPEN)
            return (False, meta)

        except Exception as e:
            try:
                metrics.stop_arm_failed_total.labels(SERVICE, exchange.name, symbol, "POLL").inc()
            except Exception:
                pass
            # polling failure doesn't change state
            return (False, meta)

    # 2) no stop id yet: try arm (recovery / previous arm failed)
    if not runtime_cfg.use_protective_stop_order:
        return (False, meta)
    if meta.get("stop_arming_disabled"):
        return (False, meta)

    last_arm_ms = int(meta.get("stop_last_arm_ms") or 0)
    if (now_ms - last_arm_ms) < int(runtime_cfg.stop_rearm_cooldown_seconds) * 1000:
        return (False, meta)

    base_open = str(meta.get("open_client_order_id") or meta.get("entry_client_order_id") or "")
    if not base_open:
        # 没有 entry 的 client_order_id：只做 best-effort，使用 trade_id 生成一个基准
        base_open = f"open_{symbol}_{trade_id}" if trade_id > 0 else f"open_{symbol}"

    meta["stop_last_arm_ms"] = now_ms

    stop_seq = int(meta.get("stop_rearm_seq") or 1)
    new_stop_cid, new_stop_eid = _arm_protective_stop_with_retry(
        exchange=exchange,
        db=db,
        metrics=metrics,
        telegram=telegram,
        settings=settings,
        runtime_cfg=runtime_cfg,
        symbol=symbol,
        qty=float(base_qty),
        stop_price=float(stop_price),
        trace_id=trace_id,
        trade_id=int(trade_id),
        base_open_client_order_id=base_open,
        action="ARM",
        seq=int(stop_seq),
    )
    if new_stop_cid and new_stop_eid:
        meta["stop_client_order_id"] = new_stop_cid
        meta["stop_exchange_order_id"] = new_stop_eid
    return (False, meta)


def _parse_json_maybe(s: object) -> dict:
    try:
        if s is None:
            return {}
        if isinstance(s, dict):
            return s
        if isinstance(s, str):
            s2 = s.strip()
            if not s2:
                return {}
            return json.loads(s2)
        return {}
    except Exception:
        return {}


def _vectorize_for_ai(latest: dict) -> tuple[list[float], dict]:
    """Build AI vector from market_data_cache.features_json + basic TA fields."""
    f = _parse_json_maybe(latest.get("features_json"))

    ema_fast = float(latest.get("ema_fast") or 0.0)
    ema_slow = float(latest.get("ema_slow") or 0.0)
    rsi = float(latest.get("rsi") or 50.0)

    def _g(key: str, default: float = 0.0) -> float:
        try:
            v = f.get(key)
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    x = [
        ema_fast,
        ema_slow,
        rsi,
        _g("atr14"),
        _g("adx14"),
        _g("plus_di14"),
        _g("minus_di14"),
        _g("bb_width20"),
        _g("vol_ratio"),
        _g("mom10"),
        _g("ret1"),
        _g("ret_std20"),
    ]
    bundle = {"ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi, "features": f, "x": x}
    return x, bundle


def _load_ai_model(db: MariaDB, settings: Settings):
    """加载 AI 模型（支持 online_lr / sgd_compat）。"""
    dim = 12
    impl = (settings.ai_model_impl or 'online_lr').strip().lower()

    # Prefer ai_models(is_current=1)
    blob = load_current_model_blob(db, model_name=settings.ai_model_key)
    if blob:
        try:
            stored_impl = str(blob.get('impl') or '').strip().lower()
            if stored_impl == 'sgd_compat':
                return SGDClassifierCompat.from_dict(blob, fallback_dim=dim)
            return OnlineLogisticRegression.from_dict(blob, fallback_dim=dim)
        except Exception:
            pass

    raw = get_system_config(db, settings.ai_model_key, default='')

    if raw:
        try:
            d = json.loads(raw)
            stored_impl = str(d.get("impl") or "").strip().lower()
            if stored_impl == "sgd_compat":
                return SGDClassifierCompat.from_dict(d, fallback_dim=dim)
            # default
            return OnlineLogisticRegression.from_dict(d, fallback_dim=dim)
        except Exception:
            pass

    if impl == "sgd_compat":
        return SGDClassifierCompat(dim=dim, lr=float(settings.ai_lr), l2=float(settings.ai_l2))
    return OnlineLogisticRegression(dim=dim, lr=float(settings.ai_lr), l2=float(settings.ai_l2))


def _maybe_persist_ai_model(db: MariaDB, settings: Settings, model, *, trace_id: str, force: bool = False) -> None:
    # Persist every 10 updates or on force.
    if not force and (int(model.seen) % 10) != 0:
        return
    try:
        # Persist to ai_models as current (best-effort)
        try:
            save_current_model_blob(db, model_name=settings.ai_model_key, version='iter8', model_dict=model.to_dict(), metrics={'seen': int(getattr(model,'seen',0))})
        except Exception:
            pass

        write_system_config(
            db,
            actor="strategy-engine",
            key=settings.ai_model_key,
            value=json.dumps(model.to_dict(), ensure_ascii=False),
            trace_id=trace_id,
            reason_code=ReasonCode.AI_TRAIN.value,
            reason=f"AI model updated seen={model.seen}",
            action="AI_MODEL_UPDATE",
        )
    except Exception:
        return


def _open_trade_log(
    db: MariaDB,
    *,
    trace_id: str,
    symbol: str,
    qty: float,
    actor: str,
    leverage: int,
    stop_dist_pct: float,
    stop_price: float,
    client_order_id: str,
    robot_score: float,
    ai_prob: float | None,
    open_reason_code: str,
    open_reason: str,
    features_bundle: dict,
) -> int:
    now_ms_i = int(time.time() * 1000)
    payload = dict(features_bundle or {})
    payload.update({"robot_score": robot_score, "ai_prob": ai_prob})
    with db.tx() as cur:
        cur.execute(
            """
            INSERT INTO trade_logs(
              trace_id, actor, symbol, side, qty, leverage, stop_dist_pct, stop_price, client_order_id,
              robot_score, ai_prob, open_reason_code, open_reason, entry_time_ms, features_json, status
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                trace_id,
                actor,
                symbol,
                "BUY",
                float(qty),
                int(leverage),
                float(stop_dist_pct),
                float(stop_price),
                client_order_id,
                float(robot_score),
                float(ai_prob) if ai_prob is not None else None,
                open_reason_code,
                open_reason,
                now_ms_i,
                json.dumps(payload, ensure_ascii=False),
                "OPEN",
            ),
        )
        return int(cur.lastrowid or 0)




def _fetch_trade_stop_order(db: MariaDB, trade_id: int) -> dict:
    row = db.fetch_one(
        """
        SELECT stop_client_order_id, stop_exchange_order_id, stop_order_type
        FROM trade_logs
        WHERE id=%s
        """,
        (int(trade_id),),
    ) or {}
    return {
        "stop_client_order_id": row.get("stop_client_order_id"),
        "stop_exchange_order_id": row.get("stop_exchange_order_id"),
        "stop_order_type": row.get("stop_order_type"),
    }


def _update_trade_stop_order(
    db: MariaDB,
    *,
    trade_id: int,
    stop_client_order_id: str | None,
    stop_exchange_order_id: str | None,
    stop_order_type: str | None,
) -> None:
    with db.tx() as cur:
        cur.execute(
            """UPDATE trade_logs SET stop_client_order_id=%s, stop_exchange_order_id=%s, stop_order_type=%s WHERE id=%s""",
            (stop_client_order_id, stop_exchange_order_id, stop_order_type, int(trade_id)),
        )


def _update_trade_after_entry_fill(
    db: MariaDB,
    *,
    trade_id: int,
    entry_price: float | None,
    exchange_order_id: str | None,
    stop_price: float | None,
) -> None:
    with db.tx() as cur:
        cur.execute(
            """UPDATE trade_logs SET entry_price=%s, exchange_order_id=%s, stop_price=%s WHERE id=%s""",
            (
                float(entry_price) if entry_price is not None else None,
                exchange_order_id,
                float(stop_price) if stop_price is not None else None,
                int(trade_id),
            ),
        )


def _find_open_trade_id(db: MariaDB, symbol: str, meta: dict) -> int:
    try:
        tid = int(meta.get("trade_id") or 0)
        if tid > 0:
            return tid
    except Exception:
        pass
    row = db.fetch_one("SELECT id FROM trade_logs WHERE symbol=%s AND status='OPEN' ORDER BY id DESC LIMIT 1", (symbol,))
    return int(row["id"]) if row else 0


def _close_trade_and_train(
    db: MariaDB,
    settings: Settings,
    metrics: Metrics,
    model,
    *,
    trade_id: int,
    symbol: str,
    qty: float,
    exit_price: float | None,
    pnl_usdt: float | None,
    close_reason_code: str,
    close_reason: str,
    trace_id: str,
) -> None:
    now_ms_i = int(time.time() * 1000)
    row = db.fetch_one("SELECT entry_price, entry_time_ms, features_json FROM trade_logs WHERE id=%s", (int(trade_id),))
    entry_price = float(row["entry_price"]) if row and row.get("entry_price") is not None else None

    if pnl_usdt is None and exit_price is not None and entry_price is not None:
        pnl_usdt = (float(exit_price) - float(entry_price)) * float(qty)

    label = None
    if pnl_usdt is not None:
        label = 1 if float(pnl_usdt) > 0 else 0

    with db.tx() as cur:
        cur.execute(
            """
            UPDATE trade_logs
            SET exit_price=%s, pnl=%s, close_reason_code=%s, close_reason=%s, exit_time_ms=%s, label=%s, status='CLOSED'
            WHERE id=%s
            """,
            (
                float(exit_price) if exit_price is not None else None,
                float(pnl_usdt) if pnl_usdt is not None else None,
                close_reason_code,
                close_reason,
                now_ms_i,
                int(label) if label is not None else None,
                int(trade_id),
            ),
        )

    metrics.trades_close_total.labels(SERVICE, symbol, close_reason_code).inc()
    if pnl_usdt is not None:
        metrics.trade_last_pnl_usdt.labels(SERVICE, symbol).set(float(pnl_usdt))
    if row and row.get("entry_time_ms"):
        dur = max(0.0, (now_ms_i - int(row["entry_time_ms"])) / 1000.0)
        metrics.trade_last_duration_seconds.labels(SERVICE, symbol).set(dur)

    if settings.ai_enabled and model is not None and label is not None and row and row.get("features_json"):
        try:
            fj = _parse_json_maybe(row["features_json"])
            x = fj.get("x") or []
            if isinstance(x, list) and x:
                model.partial_fit([float(v) for v in x], int(label))
                metrics.ai_training_total.labels(SERVICE, symbol).inc()
                metrics.ai_model_seen.labels(SERVICE).set(int(model.seen))
                _maybe_persist_ai_model(db, settings, model, trace_id=trace_id)
        except Exception:
            pass

def setup_b_decision(
    latest: dict,
    prev: dict | None,
    *,
    ai_score: float,
    settings: Settings,
) -> tuple[bool, ReasonCode, str]:
    """V8.3 Setup B decision.

    Conditions (best-effort):
      - ADX >= threshold and +DI > -DI
      - Squeeze release (prev squeeze_status==1 and latest==0)
      - Momentum flips from negative to positive (mom10)
      - Volume ratio >= threshold
      - AI score >= threshold

    Returns: (should_buy, reason_code, reason)
    """
    f = _parse_json_maybe(latest.get("features_json"))
    fp = _parse_json_maybe(prev.get("features_json")) if prev else {}

    def _fnum(d: dict, k: str):
        try:
            v = d.get(k)
            return float(v) if v is not None else None
        except Exception:
            return None

    adx = _fnum(f, "adx14")
    pdi = _fnum(f, "plus_di14")
    mdi = _fnum(f, "minus_di14")
    vol_ratio = _fnum(f, "vol_ratio")
    mom = _fnum(f, "mom10")
    sq = _fnum(f, "squeeze_status")
    mom_prev = _fnum(fp, "mom10")
    sq_prev = _fnum(fp, "squeeze_status")

    adx_min = float(getattr(settings, "setup_b_adx_min", 20))
    vol_min = float(getattr(settings, "setup_b_vol_ratio_min", 1.5))
    ai_min = float(getattr(settings, "setup_b_ai_score_min", 55))

    squeeze_release = (sq_prev == 1.0 and sq == 0.0)
    mom_flip_pos = (mom_prev is not None and mom is not None and mom_prev < 0.0 and mom > 0.0)

    ok = True
    reasons = []
    if adx is None or pdi is None or mdi is None:
        ok = False
        reasons.append("missing_adx_di")
    else:
        if adx < adx_min:
            ok = False
            reasons.append(f"adx<{adx_min}")
        if pdi <= mdi:
            ok = False
            reasons.append("+DI<=-DI")

    if not squeeze_release:
        ok = False
        reasons.append("no_squeeze_release")
    if not mom_flip_pos:
        ok = False
        reasons.append("no_mom_flip_pos")
    if vol_ratio is None or vol_ratio < vol_min:
        ok = False
        reasons.append(f"vol_ratio<{vol_min}")
    if float(ai_score) < ai_min:
        ok = False
        reasons.append(f"ai<{ai_min}")

    reason_code = ReasonCode.SETUP_B_SQUEEZE_RELEASE
    if ok:
        reason = (
            f"Squeeze释放+动量转正+量能放大，ADX趋势确认; "
            f"adx={adx:.1f}, +di={pdi:.1f}, -di={mdi:.1f}, "
            f"vol_ratio={vol_ratio:.2f}, mom10={mom:.4f}, ai={float(ai_score):.1f}"
        )
        return True, reason_code, reason

    reason = "SetupB未满足: " + ", ".join(reasons)
    return False, reason_code, reason


def setup_b_signal(latest: dict) -> Optional[str]:
    """Backward compatible wrapper."""
    ok, _, _ = setup_b_decision(latest, None, ai_score=50.0, settings=load_settings())
    return "BUY" if ok else None
    if float(ema_fast) > float(ema_slow) and (rsi is None or float(rsi) < 70):
        return "BUY"
    if float(ema_fast) < float(ema_slow):
        return "SELL"
    return None

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_robot_score(latest: dict, *, signal: str) -> float:
    """机器人评分（0~100）。

    说明：原始需求里提到“机器人评分”，但 MVP 只缓存了 EMA/RSI。
    这里用 EMA 趋势强度 + RSI 位置来计算一个可解释的评分：
    - 趋势强：EMA_fast 与 EMA_slow 偏离越大，score 越高
    - BUY：RSI 越接近 30~55 越好；SELL：RSI 越接近 45~70 越好（更偏向超买）

    该评分用于动态杠杆（10~20 倍）。
    """
    try:
        price = float(latest.get("close_price") or 0)
        ema_fast = latest.get("ema_fast")
        ema_slow = latest.get("ema_slow")
        rsi = latest.get("rsi")

        if price <= 0 or ema_fast is None or ema_slow is None:
            return 50.0

        ema_fast_f = float(ema_fast)
        ema_slow_f = float(ema_slow)
        rsi_f = float(rsi) if rsi is not None else 50.0

        # 趋势强度：EMA 偏离百分比（例如 0.10% 就给到满 50 分）
        diff_pct = abs(ema_fast_f - ema_slow_f) / price * 100.0
        trend_score = _clamp(diff_pct * 500.0, 0.0, 50.0)

        if signal == "BUY":
            # 越接近低位越好（30 最佳，70 最差）
            rsi_score = _clamp((70.0 - rsi_f) / 40.0 * 50.0, 0.0, 50.0)
        else:
            # SELL 更偏向 RSI 高位
            rsi_score = _clamp((rsi_f - 30.0) / 40.0 * 50.0, 0.0, 50.0)

        return _clamp(trend_score + rsi_score, 0.0, 100.0)
    except Exception:
        return 50.0

def leverage_from_score(settings: Settings, score: float) -> int:
    """根据评分映射杠杆倍数（10~20）。"""
    lo = int(settings.auto_leverage_min)
    hi = int(settings.auto_leverage_max)
    if hi < lo:
        hi = lo
    s = _clamp(float(score), 0.0, 100.0)
    # 线性映射：0 -> lo, 100 -> hi
    lev = lo + int(round((hi - lo) * (s / 100.0)))
    return int(_clamp(lev, lo, hi))

class CircuitBreaker:
    def __init__(self, *, window_seconds: int, rate_limit_threshold: int, failure_threshold: int):
        self.window_seconds = int(window_seconds)
        self.rate_limit_threshold = int(rate_limit_threshold)
        self.failure_threshold = int(failure_threshold)
        self._rate_limits: list[float] = []
        self._failures: list[float] = []

    def _prune(self) -> None:
        now = time.time()
        cutoff = now - float(self.window_seconds)
        self._rate_limits = [t for t in self._rate_limits if t >= cutoff]
        self._failures = [t for t in self._failures if t >= cutoff]

    def record_rate_limit(self) -> None:
        self._rate_limits.append(time.time())
        self._prune()

    def record_failure(self) -> None:
        self._failures.append(time.time())
        self._prune()

    def should_halt(self) -> tuple[bool, str]:
        self._prune()
        if len(self._rate_limits) >= self.rate_limit_threshold:
            return True, f"rate_limit_count={len(self._rate_limits)}/{self.rate_limit_threshold}"
        if len(self._failures) >= self.failure_threshold:
            return True, f"failure_count={len(self._failures)}/{self.failure_threshold}"
        return False, ""


def get_equity_usdt(exchange: ExchangeClient, settings: Settings) -> float:
    # Best-effort: use exchange capability; else env/config fallback.
    if hasattr(exchange, "get_equity_usdt"):
        try:
            v = exchange.get_equity_usdt()
            return float(v)
        except Exception:
            pass
    try:
        return float(getattr(settings, "account_equity_usdt", 0.0) or 0.0)
    except Exception:
        return 0.0


def compute_base_margin_usdt(*, equity_usdt: float, ai_score: float, settings: Settings) -> float:
    base = max(50.0, float(equity_usdt) * 0.10)
    # allow boost when ai_score high (V8.3)
    if float(ai_score) > 85.0:
        base *= 1.2
    return float(base)


def enforce_risk_budget(
    *,
    equity_usdt: float,
    base_margin_usdt: float,
    leverage: int,
    stop_dist_pct: float,
    settings: Settings,
) -> tuple[bool, int, str]:
    """Hard risk budget (V8.3).

    risk_amount ~= base_margin * leverage * stop_dist_pct
    must be <= equity * risk_budget_pct.
    If exceeded -> reduce leverage down to 1; still exceeded -> reject.
    """
    budget = float(equity_usdt) * float(getattr(settings, "risk_budget_pct", 0.03))
    lev = int(leverage)
    stop_pct = max(0.0, float(stop_dist_pct))
    if budget <= 0:
        return True, lev, "no_budget_configured"

    def risk_amt(lv: int) -> float:
        return float(base_margin_usdt) * float(lv) * float(stop_pct)

    if risk_amt(lev) <= budget:
        return True, lev, f"risk_ok risk={risk_amt(lev):.2f}<=budget={budget:.2f}"

    # reduce leverage
    while lev > 1 and risk_amt(lev) > budget:
        lev -= 1

    if risk_amt(lev) <= budget:
        return True, lev, f"risk_adjusted risk={risk_amt(lev):.2f}<=budget={budget:.2f}"

    return False, lev, f"risk_reject risk={risk_amt(lev):.2f}>budget={budget:.2f}"


def min_qty_from_min_margin_usdt(min_margin_usdt: float, last_price: float, leverage: int, *, precision: int = 6) -> float:
    """根据“每单最小保证金(min_margin_usdt)”与杠杆计算最小下单数量 qty。

    你要求的口径是：
    - 50U 是**实际保证金**（投入资金），不是名义仓位
    - 合约名义价值(notional) 约等于：保证金 * 杠杆
    - 下单名义价值(notional) 约等于：价格 * 数量

    因此最小 qty 的估算方式为：
        notional_min = min_margin_usdt * leverage
        qty_min = notional_min / last_price

    注意：交易所对 qty step/最小下单量各不相同，真实生产建议：
    - 通过交易所接口查询 symbol 的 lotSize / qtyStep
    - 根据 step 做“向上取整”

    这里先用固定小数位（默认 6 位）做“向上取整”，保证 notional >= notional_min。
    """
    import math

    if last_price <= 0:
        return 0.0
    lev = max(1, int(leverage))
    notional_min = float(min_margin_usdt) * float(lev)
    q = notional_min / float(last_price)
    if q <= 0:
        return 0.0

    factor = 10 ** int(precision)
    q_up = math.ceil(q * factor) / factor
    return float(q_up)

def get_latest_positions_map(db: MariaDB, symbols: list[str]) -> dict[str, float]:
    """获取每个 symbol 最新持仓数量（base_qty）。"""
    if not symbols:
        return {}

    # 用子查询取每个 symbol 的最新一条快照
    placeholders = ",".join(["%s"] * len(symbols))
    rows = db.fetch_all(
        f"""
        SELECT ps.symbol, ps.base_qty
        FROM position_snapshots ps
        JOIN (
            SELECT symbol, MAX(id) AS max_id
            FROM position_snapshots
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
        ) t ON t.symbol=ps.symbol AND t.max_id=ps.id
        """,
        tuple(symbols),
    )
    out: dict[str, float] = {}
    for r in rows:
        out[str(r["symbol"]).upper()] = float(r["base_qty"])
    return out

def main():
    settings = load_settings()
    exchange = settings.exchange
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, Path(__file__).resolve().parents[2] / "migrations")

    metrics = Metrics(SERVICE)
    start_metrics_http_server(int(settings.metrics_port) or 9102)
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    breaker = CircuitBreaker(
        window_seconds=int(getattr(settings, "circuit_window_seconds", 600)),
        rate_limit_threshold=int(getattr(settings, "circuit_rate_limit_threshold", 8)),
        failure_threshold=int(getattr(settings, "circuit_failure_threshold", 6)),
    )

    r = redis_client(settings.redis_url)

    instance_id = get_instance_id(settings.instance_id)
    leader_key = f"{settings.leader_key_prefix}:{SERVICE}"
    elector = LeaderElector(
        r,
        key=leader_key,
        instance_id=instance_id,
        ttl_seconds=settings.leader_ttl_seconds,
        renew_interval_seconds=settings.leader_renew_interval_seconds,
    )
    last_role: str = "unknown"

    ex = make_exchange(settings, metrics=metrics, service_name=SERVICE)

    # Control commands poller: apply NEW commands every 1-3 seconds (decoupled from tick)
    def _control_poller_loop():
        interval = float(getattr(settings, 'control_poll_seconds', 2.0) or 2.0)
        while True:
            poll_trace_id = new_trace_id('control_poll')
            try:
                apply_control_commands(db, telegram, exchange=settings.exchange, trace_id=poll_trace_id)
            except Exception as e:
                try:
                    log_action(logger, action='CONTROL_COMMANDS_POLL_ERROR', trace_id=poll_trace_id, reason_code='ERROR', reason=str(e)[:200], client_order_id=None)
                except Exception:
                    pass
            time.sleep(max(0.5, interval))

    threading.Thread(target=_control_poller_loop, daemon=True).start()

    runtime_cfg = RuntimeConfig.load(db, settings)
    symbols = list(runtime_cfg.symbols)
    try:
        metrics.runtime_config_symbols_count.labels(SERVICE).set(len(runtime_cfg.symbols))
        metrics.runtime_config_last_refresh_ms.labels(SERVICE).set(runtime_cfg.last_refresh_ms)
    except Exception:
        pass
    next_cfg_refresh_ts = time.time() + float(settings.runtime_config_refresh_seconds)
    next_stop_poll_ts = time.time() + float(max(1, int(runtime_cfg.stop_order_poll_seconds)))

    while True:
        # leader election: only leader executes trading ticks; followers only heartbeat + metrics
        is_leader = True
        if settings.leader_election_enabled:
            is_leader = elector.ensure()
        metrics.leader_is_leader.labels(SERVICE, instance_id).set(1 if is_leader else 0)

        role = "leader" if is_leader else "follower"
        if role != last_role:
            metrics.leader_changes_total.labels(SERVICE, instance_id, role).inc()
            last_role = role

        # heartbeat (liveness)
        try:
            upsert_service_status(db, service_name=SERVICE, instance_id=instance_id, status={"status": "RUNNING", "role": role, "leader": elector.get_leader() if settings.leader_election_enabled else instance_id, "symbols_count": len(symbols), "symbols_from_db": runtime_cfg.symbols_from_db, "halt_trading": bool(runtime_cfg.halt_trading), "emergency_exit": bool(runtime_cfg.emergency_exit)})
        except Exception:
            pass

        if not is_leader:
            time.sleep(settings.leader_follower_sleep_seconds)
            continue

        # Sleep until next tick, but keep refreshing runtime config so SYMBOLS/HALT/EMERGENCY can hot-reload
        sleep_s = next_tick_sleep_seconds(settings.strategy_tick_seconds)
        end_ts = time.time() + float(sleep_s)
        while time.time() < end_ts:
            if time.time() >= next_cfg_refresh_ts:
                try:
                    changes = runtime_cfg.refresh(db, settings)
                    metrics.runtime_config_refresh_total.labels(SERVICE).inc()
                    metrics.runtime_config_symbols_count.labels(SERVICE).set(len(runtime_cfg.symbols))
                    metrics.runtime_config_last_refresh_ms.labels(SERVICE).set(runtime_cfg.last_refresh_ms)
                    if "symbols" in changes:
                        symbols = list(runtime_cfg.symbols)
                        logger.info(f"runtime_config_symbols_updated symbols={symbols} symbols_from_db={runtime_cfg.symbols_from_db}")
                except Exception as e:
                    logger.warning(f"runtime_config_refresh_failed err={e}")
                next_cfg_refresh_ts = time.time() + float(settings.runtime_config_refresh_seconds)

            # periodic position snapshots (V8.3): every N seconds write a snapshot for active positions
            try:
                if time.time() >= next_snapshot_ts and symbols:
                    snap_trace_id = new_trace_id("pos_snap")
                    interval_s = float(getattr(settings, "position_snapshot_interval_seconds", 300) or 300)
                    for sym in list(symbols):
                        try:
                            pos_row = get_position(db, sym)
                            if not pos_row:
                                continue
                            base_qty = float(pos_row.get("base_qty") or 0.0)
                            if base_qty <= 0:
                                continue
                            created_at = pos_row.get("created_at")
                            # If DB time not parsed, still write best-effort every interval
                            should_write = True
                            try:
                                if created_at is not None:
                                    # MariaDB returns datetime
                                    age = (datetime.datetime.utcnow() - created_at).total_seconds()
                                    should_write = age >= max(30.0, interval_s * 0.9)
                            except Exception:
                                should_write = True
                            if not should_write:
                                continue
                            meta = _parse_json_maybe(pos_row.get("meta_json"))
                            meta["note"] = "periodic_snapshot"
                            meta["trace_id"] = snap_trace_id
                            save_position(db, sym, float(base_qty), float(pos_row.get("avg_entry_price")) if pos_row.get("avg_entry_price") is not None else None, meta)
                        except Exception:
                            continue
                    next_snapshot_ts = time.time() + interval_s
            except Exception:
                pass

            # protective stop poll (between ticks) - for crash recovery / timely stop fill detection
            try:
                if (
                    time.time() >= next_stop_poll_ts
                    and bool(runtime_cfg.use_protective_stop_order)
                    and settings.exchange != "paper"
                    and symbols
                ):
                    poll_trace_id = new_trace_id("stop_poll")
                    # Only poll symbols that currently have a position > 0
                    for sym in list(symbols):
                        try:
                            pos = get_position(db, sym)
                            base_qty = float(pos["base_qty"]) if pos else 0.0
                            avg_entry = float(pos["avg_entry_price"]) if pos and pos["avg_entry_price"] is not None else None
                            if base_qty <= 0:
                                continue

                            meta_before = _parse_json_maybe(pos.get("meta_json") if pos else None)
                            closed_by_stop, meta_after = _ensure_protective_stop(
                                exchange=ex,
                                db=db,
                                metrics=metrics,
                                telegram=telegram,
                                settings=settings,
                                runtime_cfg=runtime_cfg,
                                symbol=sym,
                                base_qty=float(base_qty),
                                avg_entry=avg_entry,
                                pos_row=pos,
                                trace_id=poll_trace_id,
                            )
                            if meta_after != meta_before:
                                save_position(
                                    db,
                                    sym,
                                    float(base_qty),
                                    float(avg_entry) if avg_entry is not None else None,
                                    meta_after,
                                )

                            if closed_by_stop:
                                stop_fill = meta_after.pop("_stop_fill", {}) if isinstance(meta_after, dict) else {}
                                exit_price = stop_fill.get("avg_price") or meta_after.get("stop_price") or avg_entry
                                trade_id2 = _find_open_trade_id(db, sym, meta_after if isinstance(meta_after, dict) else {})
                                save_position(
                                    db,
                                    sym,
                                    0.0,
                                    None,
                                    {"trace_id": poll_trace_id, "note": "protective_stop_filled", "trade_id": trade_id2},
                                )
                                if trade_id2 > 0:
                                    _close_trade_and_train(
                                        db,
                                        settings,
                                        metrics,
                                        _load_ai_model(db, settings) if settings.ai_enabled else None,
                                        trade_id=trade_id2,
                                        symbol=sym,
                                        qty=float(base_qty),
                                        exit_price=float(exit_price) if exit_price is not None else None,
                                        pnl_usdt=stop_fill.get("pnl_usdt"),
                                        close_reason_code=ReasonCode.STOP_LOSS.value,
                                        close_reason="Protective stop filled (exchange)",
                                        trace_id=poll_trace_id,
                                    )
                                summary_kv = build_trade_summary(
                                    event="PROTECTIVE_STOP_FILLED",
                                    trace_id=poll_trace_id,
                                    exchange=exchange,
                                    symbol=sym,
                                    side="SELL",
                                    qty=base_qty,
                                    price=exit_price,
                                    stop_price=None,
                                    reason_code=ReasonCode.STOP_LOSS,
                                    reason="Protective stop filled (exchange)",
                                    extra={"trade_id": str(trade_id2)},
                                )
                                send_trade_alert(telegram, title="交易所止损单成交", summary_kv=summary_kv, payload={})
                                log_action(
                                    logger,
                                    "PROTECTIVE_STOP_FILLED",
                                    trace_id=poll_trace_id,
                                    exchange=exchange,
                                    symbol=sym,
                                    side="SELL",
                                    qty=base_qty,
                                    price=exit_price,
                                    reason_code=ReasonCode.STOP_LOSS.value,
                                    reason="Protective stop filled (exchange)",
                                )
                        except RateLimitError:
                            breaker.record_rate_limit()
                            # avoid hammering exchange during stop polling
                            break
                        except Exception:
                            pass
                    next_stop_poll_ts = time.time() + float(max(1, int(runtime_cfg.stop_order_poll_seconds)))
            except Exception:
                pass

            time.sleep(min(2.0, max(0.1, end_ts - time.time())))

        trace_id = new_trace_id("tick")

        try:
            # HALT_TRADING: hot-reload from system_config
            if bool(runtime_cfg.halt_trading):
                telegram.send(f"[HALT] 本轮跳过 trace_id={trace_id} symbols={','.join(symbols)}")
                continue

            tick_id = int(time.time() // settings.strategy_tick_seconds)

            tick_start_ts = time.time()
            def _budget_exceeded() -> bool:
                return (time.time() - float(tick_start_ts)) > float(settings.tick_budget_seconds)

            # best-effort reconcile (stale CREATED/SUBMITTED orders)
            try:
                fixed = reconcile_stale_orders(db, ex, exchange_name=settings.exchange, max_age_seconds=180, metrics=metrics, telegram=telegram)
                if fixed:
                    logger.info(f"reconcile_fixed={fixed} trace_id={trace_id}")
            except Exception:
                pass

            # 先计算“当前全局已持仓数量”，用于限制最多 3 单（跨交易对）
            pos_map = get_latest_positions_map(db, tuple(symbols))
            open_cnt = sum(1 for q in pos_map.values() if q > 0)

            # --- AI 选币：从 SYMBOLS(10-20) 中选择“最优”开仓币对 ---
            # 需求：同一时间最多只允许 MAX_CONCURRENT_POSITIONS 个仓位（跨交易对全局限制）。
            # 我们对“当前无持仓”的币对计算 BUY 信号与机器人评分，并按评分排序，取前 N 个执行开仓。
            selected_open_symbols: set[str] = set()
            selected_open_meta: dict[str, dict] = {}
            try:
                max_pos = int(settings.max_concurrent_positions)
                available_slots = max(0, max_pos - open_cnt)
                if available_slots > 0:
                    candidates = []  # (combined_score, symbol, meta)
                    ai_model = _load_ai_model(db, settings) if settings.ai_enabled else None
                    for s in symbols:
                        if _budget_exceeded():
                            log_action(logger, "TICK_TIME_BUDGET_EXCEEDED", trace_id=trace_id, reason_code=ReasonCode.TICK_TIMEOUT.value, reason=f"tick budget exceeded during selection (>{settings.tick_budget_seconds}s)")
                            break
                        if float(pos_map.get(s, 0.0) or 0.0) > 0.0:
                            continue  # 已有仓位的币对不参与“选币开仓”，但仍会参与后续平仓/止损逻辑
                        latest_s, prev_s = last_two_cache(db, s, settings.interval_minutes, settings.feature_version)
                        if not latest_s:
                            continue
                        # 你要求的口径：MIN_ORDER_USDT 是“实际保证金(USDT)”，名义价值 = 价格*qty ≈ 保证金*杠杆。
                        # 因此选币阶段也要用“保证金*杠杆”的方式反推 qty，避免选中后又因 qty 过小被跳过。
                        try:
                            last_px = float(latest_s.get("close_price") or 0.0)
                        except Exception:
                            last_px = 0.0
                        if last_px <= 0:
                            continue
                        score_s = compute_robot_score(latest_s, signal="BUY")
                        lev_s = leverage_from_score(settings, score_s)
                        qty_s = min_qty_from_min_margin_usdt(settings.min_order_usdt, last_px, lev_s, precision=6)
                        if qty_s <= 0:
                            continue
                        ai_prob = None
                        feat_bundle = {}
                        if settings.ai_enabled and ai_model is not None:
                            try:
                                x, feat_bundle = _vectorize_for_ai(latest_s)
                                ai_prob = float(ai_model.predict_proba(x))
                                metrics.ai_predictions_total.labels(SERVICE, s).inc()
                            except Exception:
                                ai_prob = None
                        combined = float(score_s)
                        if ai_prob is not None:
                            w = _clamp(float(settings.ai_weight), 0.0, 1.0)
                            combined = (1.0 - w) * float(score_s) + w * (ai_prob * 100.0)
                        # V8.3 Setup B: decision must include AI score + prev bar for squeeze/mom flip
                        ai_score_s = float(ai_prob * 100.0) if ai_prob is not None else 50.0
                        should_buy_s, open_reason_code_s, open_reason_s = setup_b_decision(
                            latest_s,
                            prev_s,
                            ai_score=ai_score_s,
                            settings=settings,
                        )
                        if not should_buy_s:
                            continue
                        meta = {
                            "robot_score": float(score_s),
                            "ai_prob": ai_prob,
                            "combined_score": float(combined),
                            "features_bundle": feat_bundle,
                            "open_reason_code": open_reason_code_s.value,
                            "open_reason": open_reason_s,
                        }
                        candidates.append((float(combined), s, meta))
                    # 按评分从高到低选择前 N 个开仓候选
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    selected_open_symbols = set([sym for _, sym, _ in candidates[:available_slots]])
                    selected_open_meta = {sym: meta for _, sym, meta in candidates[:available_slots]}
            except Exception:
                # 选币失败不应导致主循环崩溃：回退为“无候选”，本轮不主动开仓
                selected_open_symbols = set()
                selected_open_meta = {}
            apply_control_commands(db, telegram, exchange=settings.exchange, trace_id=trace_id)
            # Circuit breaker auto-halt
            should_halt, note = breaker.should_halt()
            if should_halt and get_flag(db, "HALT_TRADING", "false") != "true":
                set_flag(db, "HALT_TRADING", "true")
                send_system_alert(
                    telegram,
                    title="熔断触发暂停交易",
                    summary_kv=build_system_summary(
                        event="CIRCUIT_HALT",
                        trace_id=trace_id,
                        exchange=exchange,
                        level="WARN",
                        reason_code=ReasonCode.CIRCUIT_BREAKER_HALT,
                        reason=note,
                    ),
                    payload={"note": note},
                )
                log_action(
                    logger,
                    action="CIRCUIT_BREAKER_HALT",
                    trace_id=trace_id,
                    reason_code=ReasonCode.CIRCUIT_BREAKER_HALT,
                    reason=note,
                    client_order_id=None,
                )
                append_order_event(db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol="*",
                                 client_order_id=None, exchange_order_id=None, event_type=OrderEventType.ERROR,
                                 side=Side.BUY.value, qty=0.0, price=None, status="HALT",
                                 reason_code=ReasonCode.CIRCUIT_BREAKER_HALT.value, reason=note, payload={"window_seconds": breaker.window_seconds})

                log_action(logger, "CIRCUIT_BREAKER_HALT", trace_id=trace_id,
                           reason_code=ReasonCode.CIRCUIT_BREAKER_HALT.value, reason=note,
                           window_seconds=breaker.window_seconds)

            for symbol in symbols:
                if _budget_exceeded():
                    log_action(logger, "TICK_TIME_BUDGET_EXCEEDED", trace_id=trace_id, reason_code=ReasonCode.TICK_TIMEOUT.value, reason=f"tick budget exceeded before processing symbols (>{settings.tick_budget_seconds}s)")
                    break
                lock_key = f"asv8:lock:trade:{symbol}"
                with distributed_lock(r, lock_key, ttl_ms=int(min(float(settings.trade_lock_ttl_seconds), float(settings.strategy_tick_seconds)) * 1000)) as acquired:
                    if not acquired:
                        continue

                    latest, prev = last_two_cache(db, symbol, settings.interval_minutes, settings.feature_version)
                    if not latest:
                        continue

                    last_price = float(latest["close_price"])
                    if hasattr(ex, "update_last_price"):
                        ex.update_last_price(symbol, last_price)

                    pos = get_position(db, symbol)
                    base_qty = float(pos["base_qty"]) if pos else 0.0
                    avg_entry = float(pos["avg_entry_price"]) if pos and pos["avg_entry_price"] is not None else None

                    # --- 交易所保护止损单：轮询 + 异常分支（拒绝/过期/取消）自动重挂或降级 ---
                    if base_qty > 0 and bool(runtime_cfg.use_protective_stop_order) and settings.exchange != "paper":
                        meta_before = _parse_json_maybe(pos.get("meta_json") if pos else None)
                        closed_by_stop, meta_after = _ensure_protective_stop(
                            exchange=ex,
                            db=db,
                            metrics=metrics,
                            telegram=telegram,
                            settings=settings,
                            runtime_cfg=runtime_cfg,
                            symbol=symbol,
                            base_qty=float(base_qty),
                            avg_entry=avg_entry,
                            pos_row=pos,
                            trace_id=trace_id,
                        )
                        # 仅当 meta 有变化时写快照（避免每轮都写）
                        try:
                            if meta_after != meta_before:
                                save_position(db, symbol, float(base_qty), float(avg_entry) if avg_entry is not None else None, meta_after)
                        except Exception:
                            pass

                        if closed_by_stop:
                            # 止损单已成交 -> 更新本地仓位为 0，关闭 trade
                            stop_fill = meta_after.pop("_stop_fill", {}) if isinstance(meta_after, dict) else {}
                            exit_price = stop_fill.get("avg_price") or meta_after.get("stop_price") or avg_entry
                            trade_id2 = _find_open_trade_id(db, symbol, meta_after if isinstance(meta_after, dict) else {})
                            save_position(db, symbol, 0.0, None, {"trace_id": trace_id, "note": "protective_stop_filled", "trade_id": trade_id2})
                            if trade_id2 > 0:
                                _close_trade_and_train(
                                    db,
                                    settings,
                                    metrics,
                                    _load_ai_model(db, settings) if settings.ai_enabled else None,
                                    trade_id=trade_id2,
                                    symbol=symbol,
                                    qty=float(base_qty),
                                    exit_price=float(exit_price) if exit_price is not None else None,
                                    pnl_usdt=stop_fill.get("pnl_usdt"),
                                    close_reason_code=ReasonCode.STOP_LOSS.value,
                                    close_reason="Protective stop filled (exchange)",
                                    trace_id=trace_id,
                                )
                            open_cnt = max(0, open_cnt - 1)
                            stop_p = None
                            try:
                                stop_p = float(meta_after.get("stop_price") or 0.0) if isinstance(meta_after, dict) else None
                            except Exception:
                                stop_p = None
                            summary_kv = build_trade_summary(
                                event="PROTECTIVE_STOP_FILLED",
                                trace_id=trace_id,
                                exchange=exchange,
                                symbol=symbol,
                                side="SELL",
                                qty=base_qty,
                                price=exit_price,
                                stop_price=stop_p,
                                reason_code=ReasonCode.STOP_LOSS,
                                reason="Protective stop filled (exchange)",
                                extra={"pnl_usdt": stop_fill.get("pnl_usdt")},
                            )
                            send_trade_alert(
                                telegram,
                                title="交易所止损单成交",
                                summary_kv=summary_kv,
                                payload={"stop_fill": stop_fill},
                            )
                            log_action(
                                logger,
                                "PROTECTIVE_STOP_FILLED",
                                trace_id=trace_id,
                                exchange=exchange,
                                symbol=symbol,
                                side="SELL",
                                qty=base_qty,
                                price=exit_price,
                                stop_price=stop_p,
                                reason_code=ReasonCode.STOP_LOSS.value,
                                reason="Protective stop filled (exchange)",
                                pnl_usdt=stop_fill.get("pnl_usdt"),
                            )
                            try:
                                metrics.orders_total.labels(SERVICE, settings.exchange, symbol, "STOP_LOSS").inc()
                            except Exception:
                                pass
                            continue

                    # --- 紧急退出：对所有交易对生效 ---
                    if bool(runtime_cfg.emergency_exit):
                        if base_qty > 0:
                            client_order_id = make_client_order_id(
                                "exit",
                                symbol,
                                interval_minutes=settings.interval_minutes,
                                kline_open_time_ms=int(latest["open_time_ms"]),
                                trace_id=trace_id,
                            )
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                                side=Side.SELL.value, qty=base_qty, price=None, status="CREATED",
                                reason_code=ReasonCode.EMERGENCY_EXIT, reason="Emergency exit requested", payload={}
                            )
                            meta = _parse_json_maybe(pos.get('meta_json') if pos else None)
                            meta["base_qty"] = float(base_qty)
                            if runtime_cfg.use_protective_stop_order and settings.exchange != "paper" and meta.get("stop_client_order_id"):
                                _cancel_protective_stop(exchange=ex, db=db, symbol=symbol, trace_id=trace_id, meta=meta)
                            res = ex.place_market_order(symbol=symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                                event_type=_event_type_from_status(res.status),
                                side=Side.SELL.value, qty=base_qty, price=res.avg_price, status=res.status,
                                reason_code=ReasonCode.EMERGENCY_EXIT, reason="Emergency exit executed", payload=res.raw or {}
                            )
                            meta2 = _parse_json_maybe(pos.get("meta_json") if pos else None)
                            trade_id2 = _find_open_trade_id(db, symbol, meta2)
                            save_position(db, symbol, 0.0, None, {"trace_id": trace_id, "note": "emergency_exit", "trade_id": trade_id2})
                            if trade_id2 > 0:
                                _close_trade_and_train(
                                    db,
                                    settings,
                                    metrics,
                                    _load_ai_model(db, settings) if settings.ai_enabled else None,
                                    trade_id=trade_id2,
                                    symbol=symbol,
                                    qty=float(base_qty),
                                    exit_price=res.avg_price,
                                    pnl_usdt=res.pnl_usdt,
                                    close_reason_code=ReasonCode.EMERGENCY_EXIT.value,
                                    close_reason="Emergency exit executed",
                                    trace_id=trace_id,
                                )
                            open_cnt = max(0, open_cnt - 1)
                            send_system_alert(
                                telegram,
                                title="紧急退出已执行",
                                summary_kv=build_system_summary(
                                    event="EMERGENCY_EXIT_EXECUTED",
                                    trace_id=trace_id,
                                    exchange=exchange,
                                    level="INFO",
                                    extra={
                                        "symbol": symbol,
                                        "side": "SELL",
                                        "qty": base_qty,
                                        "price": res.avg_price,
                                        "fee_usdt": res.fee_usdt,
                                        "pnl_usdt": res.pnl_usdt,
                                    },
                                ),
                                payload={
                                    "client_order_id": client_order_id,
                                    "exchange_order_id": res.exchange_order_id,
                                    "avg_price": res.avg_price,
                                    "fee_usdt": res.fee_usdt,
                                    "pnl_usdt": res.pnl_usdt,
                                },
                            )

                        # 当所有币对都检查完后再清掉 EMERGENCY_EXIT
                        continue

                    # --- 硬止损（逐仓合约） ---
                    if base_qty > 0 and avg_entry is not None:
                        meta = _parse_json_maybe(pos.get('meta_json') if pos else None)
                        stop_dist_pct = float(meta.get('stop_dist_pct') or settings.hard_stop_loss_pct)
                        stop_price = float(meta.get('stop_price') or (avg_entry * (1.0 - stop_dist_pct)))
                        if last_price <= stop_price and not (runtime_cfg.use_protective_stop_order and settings.exchange != "paper" and meta.get('stop_client_order_id')):
                            client_order_id = make_client_order_id(
                                "sl",
                                symbol,
                                interval_minutes=settings.interval_minutes,
                                kline_open_time_ms=int(latest["open_time_ms"]),
                                trace_id=trace_id,
                            )
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                                side=Side.SELL.value, qty=base_qty, price=None, status="CREATED",
                                reason_code=ReasonCode.STOP_LOSS,
                                reason=f"Hard stop loss: last={last_price} <= stop={stop_price}",
                                payload={"last_price": last_price, "stop_price": stop_price}
                            )
                            res = ex.place_market_order(symbol=symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                                event_type=_event_type_from_status(res.status),
                                side=Side.SELL.value, qty=base_qty, price=res.avg_price, status=res.status,
                                reason_code=ReasonCode.STOP_LOSS, reason="Stop loss executed", payload=res.raw or {}
                            )
                            meta2 = _parse_json_maybe(pos.get("meta_json") if pos else None)
                            trade_id2 = _find_open_trade_id(db, symbol, meta2)
                            save_position(db, symbol, 0.0, None, {"trace_id": trace_id, "note": "stop_loss", "trade_id": trade_id2})
                            if trade_id2 > 0:
                                _close_trade_and_train(
                                    db,
                                    settings,
                                    metrics,
                                    _load_ai_model(db, settings) if settings.ai_enabled else None,
                                    trade_id=trade_id2,
                                    symbol=symbol,
                                    qty=float(base_qty),
                                    exit_price=res.avg_price,
                                    pnl_usdt=res.pnl_usdt,
                                    close_reason_code=ReasonCode.STOP_LOSS.value,
                                    close_reason="Hard stop loss triggered",
                                    trace_id=trace_id,
                                )
                            open_cnt = max(0, open_cnt - 1)
                            summary_kv = build_trade_summary(
                                event="STOP_LOSS",
                                trace_id=trace_id,
                                exchange=exchange,
                                symbol=symbol,
                                side="SELL",
                                qty=base_qty,
                                price=res.avg_price,
                                stop_price=stop_price,
                                reason_code=ReasonCode.STOP_LOSS,
                                reason="Stop loss triggered",
                                client_order_id=client_order_id,
                                exchange_order_id=res.exchange_order_id,
                                extra={"last_price": round(float(last_price), 4), "fee_usdt": res.fee_usdt, "pnl_usdt": res.pnl_usdt},
                            )
                            send_trade_alert(
                                telegram,
                                title="触发止损",
                                summary_kv=summary_kv,
                                payload={
                                    "client_order_id": client_order_id,
                                    "exchange_order_id": res.exchange_order_id,
                                    "avg_price": res.avg_price,
                                    "fee_usdt": res.fee_usdt,
                                    "pnl_usdt": res.pnl_usdt,
                                    "raw": res.raw or {},
                                },
                            )
                            log_action(
                                logger,
                                "STOP_LOSS",
                                trace_id=trace_id,
                                exchange=exchange,
                                symbol=symbol,
                                side="SELL",
                                qty=base_qty,
                                price=res.avg_price,
                                stop_price=stop_price,
                                client_order_id=client_order_id,
                                exchange_order_id=res.exchange_order_id,
                                reason_code=ReasonCode.STOP_LOSS.value,
                                reason="Stop loss triggered",
                                pnl_usdt=res.pnl_usdt,
                            )
                            metrics.orders_total.labels(SERVICE, settings.exchange, symbol, "STOP_LOSS").inc()
                            continue

                    sig = setup_b_signal(latest)
                    if sig == "BUY" and base_qty <= 0:
                        # 多币对选币开仓：仅允许本轮被 AI 选中的币对执行开仓
                        if symbol not in selected_open_symbols:
                            continue
                        # 全局最多 3 单（跨交易对）
                        if open_cnt >= int(settings.max_concurrent_positions):
                            continue

                        # 动态杠杆：10~20 倍（由机器人评分决定）
                        meta_open = selected_open_meta.get(symbol, {})
                        score = float(meta_open.get("robot_score") or compute_robot_score(latest, signal="BUY"))
                        ai_prob = meta_open.get("ai_prob")
                        combined_score = float(meta_open.get("combined_score") or score)
                        feat_bundle = meta_open.get("features_bundle") or {}
                        lev = leverage_from_score(settings, score)
                        # V8.3 risk budget hard-constraint
                        equity_usdt = get_equity_usdt(ex, settings)
                        ai_score = float(ai_prob * 100.0) if ai_prob is not None else 50.0
                        base_margin_usdt = compute_base_margin_usdt(equity_usdt=equity_usdt, ai_score=ai_score, settings=settings)
                        ok_risk, lev2, risk_note = enforce_risk_budget(
                            equity_usdt=equity_usdt,
                            base_margin_usdt=base_margin_usdt,
                            leverage=int(lev),
                            stop_dist_pct=float(settings.hard_stop_loss_pct),
                            settings=settings,
                        )
                        if not ok_risk:
                            append_order_event(
                                db,
                                trace_id=trace_id,
                                service=SERVICE,
                                exchange=exchange,
                                symbol=symbol,
                                client_order_id=None,
                                exchange_order_id=None,
                                event_type=OrderEventType.REJECTED,
                                side=Side.BUY.value,
                                qty=0.0,
                                price=None,
                                status="REJECTED",
                                reason_code=ReasonCode.RISK_BUDGET_REJECT.value,
                                reason=risk_note,
                                payload={"equity_usdt": equity_usdt, "base_margin_usdt": base_margin_usdt, "stop_dist_pct": float(settings.hard_stop_loss_pct), "ai_score": ai_score},
                            )

                        log_action(logger, "RISK_BUDGET_REJECT", trace_id=trace_id, symbol=symbol,
                                   reason_code=ReasonCode.RISK_BUDGET_REJECT.value, reason=risk_note,
                                   ai_score=ai_score, leverage=lev, stop_dist_pct=stop_dist_pct,
                                   client_order_id=client_order_id)
                        summary_kv = build_trade_summary(
                            event="RISK_REJECT",
                            trace_id=trace_id,
                            exchange=exchange,
                            symbol=symbol,
                            side="BUY",
                            leverage=lev,
                            ai_score=ai_score,
                            stop_dist_pct=stop_dist_pct,
                            reason_code=ReasonCode.RISK_BUDGET_REJECT,
                            reason=risk_note,
                            client_order_id=client_order_id,
                        )
                        send_trade_alert(telegram, title="开仓被风控拒绝", summary_kv=summary_kv, payload={"note": risk_note})
                        continue
                        if int(lev2) != int(lev):
                            lev = int(lev2)
                            append_order_event(
                                db,
                                trace_id=trace_id,
                                service=SERVICE,
                                exchange=exchange,
                                symbol=symbol,
                                client_order_id=None,
                                exchange_order_id=None,
                                event_type=OrderEventType.CREATED,
                                side=Side.BUY.value,
                                qty=0.0,
                                price=None,
                                status="ADJUST",
                                reason_code=ReasonCode.RISK_BUDGET_ADJUST_LEVERAGE.value,
                                reason=risk_note,
                                payload={"equity_usdt": equity_usdt, "base_margin_usdt": base_margin_usdt, "stop_dist_pct": float(settings.hard_stop_loss_pct), "ai_score": ai_score},
                            )

                        log_action(logger, "RISK_BUDGET_ADJUST", trace_id=trace_id, symbol=symbol,
                                   reason_code=ReasonCode.RISK_BUDGET_ADJUST_LEVERAGE.value, reason=risk_note,
                                   ai_score=ai_score, leverage_before=lev, leverage_after=lev2,
                                   stop_dist_pct=stop_dist_pct, client_order_id=client_order_id)


                        # 你要求的口径：MIN_ORDER_USDT 是“实际保证金(USDT)”，而不是名义仓位。
                        # 名义价值(notional) ≈ 价格 * qty ≈ 保证金 * 杠杆。
                        # 因此最小下单 qty 需要按 notional_min = min_margin * leverage 反推。
                        qty = min_qty_from_min_margin_usdt(settings.min_order_usdt, last_price, lev, precision=6)
                        if qty <= 0:
                            continue

                        # 设置逐仓杠杆（Bybit / Binance 合约）
                        if hasattr(ex, "set_leverage_and_margin_mode"):
                            ex.set_leverage_and_margin_mode(symbol=symbol, leverage=lev)

                        client_order_id = make_client_order_id(
                            "buy",
                            symbol,
                            interval_minutes=settings.interval_minutes,
                            kline_open_time_ms=int(latest["open_time_ms"]),
                            trace_id=trace_id,
                        )

                        # V8.3 Setup B decision (needs prev cache for squeeze/mom flip)
                        ai_score = float(ai_prob * 100.0) if ai_prob is not None else 50.0
                        should_buy, open_reason_code, open_reason = setup_b_decision(
                            latest,
                            prev,
                            ai_score=ai_score,
                            settings=settings,
                        )
                        if not should_buy:
                            continue

                        stop_dist_pct = float(settings.hard_stop_loss_pct)

                        stop_price_init = float(last_price) * (1.0 - stop_dist_pct)
                        open_reason = f"Setup B BUY; robot={round(float(score),2)}; ai_prob={round(float(ai_prob),4) if ai_prob is not None else None}; combined={round(float(combined_score),2)}"

                        trade_id = _open_trade_log(
                            db,
                            trace_id=trace_id,
                            symbol=symbol,
                            qty=float(qty),
                            actor=SERVICE,
                            leverage=int(lev),
                            stop_dist_pct=float(stop_dist_pct),
                            stop_price=float(stop_price_init),
                            client_order_id=client_order_id,
                            robot_score=float(score),
                            ai_prob=float(ai_prob) if ai_prob is not None else None,
                            open_reason_code=open_reason_code.value,
                            open_reason=open_reason,
                            features_bundle=feat_bundle if isinstance(feat_bundle, dict) else {},
                        )
                        metrics.trades_open_total.labels(SERVICE, symbol).inc()

                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                            side=Side.BUY.value, qty=qty, price=None, status="CREATED",
                            reason_code=open_reason_code,
                            reason=open_reason,
                            payload={
                                "latest": latest,
                                "robot_score": score,
                                "ai_prob": ai_prob,
                                "combined_score": combined_score,
                                "trade_id": trade_id,
                                "stop_dist_pct": stop_dist_pct,
                                "stop_price": stop_price_init,
                                "leverage": lev,
                                "min_margin_usdt": settings.min_order_usdt,
                                "notional_min_usdt": round(float(settings.min_order_usdt) * float(lev), 4),
                                "qty": qty,
                                "last_price": last_price,
                            }
                        )
                        res = ex.place_market_order(symbol=symbol, side="BUY", qty=qty, client_order_id=client_order_id)
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                            event_type=_event_type_from_status(res.status),
                            side=Side.BUY.value, qty=qty, price=res.avg_price, status=res.status,
                            reason_code=open_reason_code, reason=open_reason, payload={"exchange": res.raw or {}, "open_reason": open_reason, "open_reason_code": open_reason_code.value}

                        )
                        entry_price = res.avg_price if res.avg_price is not None else last_price
                        stop_price_final = float(entry_price) * (1.0 - float(stop_dist_pct))
                        _update_trade_after_entry_fill(
                            db,
                            trade_id=int(trade_id),
                            entry_price=float(entry_price) if entry_price is not None else None,
                            exchange_order_id=res.exchange_order_id,
                            stop_price=float(stop_price_final),
                        )
                        stop_client_order_id = None
                        stop_exchange_order_id = None
                        if runtime_cfg.use_protective_stop_order and settings.exchange != "paper":
                            stop_client_order_id, stop_exchange_order_id = _arm_protective_stop_with_retry(
                                exchange=ex,
                                db=db,
                                metrics=metrics,
                                telegram=telegram,
                                settings=settings,
                                runtime_cfg=runtime_cfg,
                                symbol=symbol,
                                qty=float(qty),
                                stop_price=float(stop_price_final),
                                trace_id=trace_id,
                                trade_id=int(trade_id),
                                base_open_client_order_id=client_order_id,
                                action="ARM",
                                seq=1,
                            )
                        meta_enter = {
                            "trace_id": trace_id,
                            "note": "entered",
                            "robot_score": float(score),
                            "leverage": int(lev),
                            "trade_id": int(trade_id),
                            "open_client_order_id": client_order_id,
                            "entry_client_order_id": client_order_id,
                            "stop_dist_pct": float(stop_dist_pct),
                            "stop_price": float(stop_price_final),
                            "stop_client_order_id": stop_client_order_id,
                            "stop_exchange_order_id": stop_exchange_order_id,
                        }
                        save_position(db, symbol, float(qty), float(entry_price), meta_enter)
                        open_cnt += 1

                        summary_kv = build_trade_summary(
                            event="BUY_FILLED",
                            trace_id=trace_id,
                            exchange=exchange,
                            symbol=symbol,
                            side="BUY",
                            qty=qty,
                            price=entry_price,
                            leverage=lev,
                            ai_score=ai_score,
                            stop_price=stop_price_final,
                            stop_dist_pct=stop_dist_pct,
                            reason_code=open_reason_code,
                            reason=open_reason,
                            client_order_id=client_order_id,
                            exchange_order_id=res.exchange_order_id,
                            stop_client_order_id=stop_client_order_id,
                            stop_exchange_order_id=stop_exchange_order_id,
                        )
                        send_trade_alert(
                            telegram,
                            title="开仓成交",
                            summary_kv=summary_kv,
                            payload={
                                "client_order_id": client_order_id,
                                "exchange_order_id": res.exchange_order_id,
                                "avg_price": res.avg_price,
                                "fee_usdt": res.fee_usdt,
                                "pnl_usdt": res.pnl_usdt,
                                "robot_score": score,
                                "leverage": lev,
                                "reason_code": open_reason_code.value if hasattr(open_reason_code, 'value') else str(open_reason_code),
                                "reason": open_reason,
                                "raw": res.raw or {},
                            },
                        )
                        log_action(
                            logger,
                            "BUY_FILLED",
                            trace_id=trace_id,
                            exchange=exchange,
                            symbol=symbol,
                            side="BUY",
                            qty=qty,
                            price=entry_price,
                            leverage=lev,
                            ai_score=ai_score,
                            stop_price=stop_price_final,
                            stop_dist_pct=stop_dist_pct,
                            client_order_id=client_order_id,
                            exchange_order_id=res.exchange_order_id,
                            reason_code=open_reason_code.value if hasattr(open_reason_code, 'value') else str(open_reason_code),
                            reason=open_reason,
                        )
                        metrics.orders_total.labels(SERVICE, settings.exchange, symbol, "BUY").inc()

                    elif sig == "SELL" and base_qty > 0:
                        meta = _parse_json_maybe(pos.get('meta_json') if pos else None)
                        meta["base_qty"] = float(base_qty)
                        if runtime_cfg.use_protective_stop_order and settings.exchange != "paper" and meta.get("stop_client_order_id"):
                            _cancel_protective_stop(exchange=ex, db=db, symbol=symbol, trace_id=trace_id, meta=meta)
                        qty = base_qty
                        score = compute_robot_score(latest, signal="SELL")
                        lev = leverage_from_score(settings, score)
                        # V8.3 risk budget hard-constraint
                        equity_usdt = get_equity_usdt(ex, settings)
                        ai_score = float(ai_prob * 100.0) if ai_prob is not None else 50.0
                        base_margin_usdt = compute_base_margin_usdt(equity_usdt=equity_usdt, ai_score=ai_score, settings=settings)
                        ok_risk, lev2, risk_note = enforce_risk_budget(
                            equity_usdt=equity_usdt,
                            base_margin_usdt=base_margin_usdt,
                            leverage=int(lev),
                            stop_dist_pct=float(settings.hard_stop_loss_pct),
                            settings=settings,
                        )
                        if not ok_risk:
                            append_order_event(
                                db,
                                trace_id=trace_id,
                                service=SERVICE,
                                exchange=exchange,
                                symbol=symbol,
                                client_order_id=None,
                                exchange_order_id=None,
                                event_type=OrderEventType.REJECTED,
                                side=Side.BUY.value,
                                qty=0.0,
                                price=None,
                                status="REJECTED",
                                reason_code=ReasonCode.RISK_BUDGET_REJECT.value,
                                reason=risk_note,
                                payload={"equity_usdt": equity_usdt, "base_margin_usdt": base_margin_usdt, "stop_dist_pct": float(settings.hard_stop_loss_pct), "ai_score": ai_score},
                            )

                        log_action(logger, "RISK_BUDGET_REJECT", trace_id=trace_id, symbol=symbol,
                                   reason_code=ReasonCode.RISK_BUDGET_REJECT.value, reason=risk_note,
                                   ai_score=ai_score, leverage=lev, stop_dist_pct=stop_dist_pct,
                                   client_order_id=client_order_id)
                        summary_kv = build_trade_summary(
                            event="RISK_REJECT",
                            trace_id=trace_id,
                            exchange=exchange,
                            symbol=symbol,
                            side="BUY",
                            leverage=lev,
                            ai_score=ai_score,
                            stop_dist_pct=stop_dist_pct,
                            reason_code=ReasonCode.RISK_BUDGET_REJECT,
                            reason=risk_note,
                            client_order_id=client_order_id,
                        )
                        send_trade_alert(telegram, title="开仓被风控拒绝", summary_kv=summary_kv, payload={"note": risk_note})
                        continue
                        if int(lev2) != int(lev):
                            lev = int(lev2)
                            append_order_event(
                                db,
                                trace_id=trace_id,
                                service=SERVICE,
                                exchange=exchange,
                                symbol=symbol,
                                client_order_id=None,
                                exchange_order_id=None,
                                event_type=OrderEventType.CREATED,
                                side=Side.BUY.value,
                                qty=0.0,
                                price=None,
                                status="ADJUST",
                                reason_code=ReasonCode.RISK_BUDGET_ADJUST_LEVERAGE.value,
                                reason=risk_note,
                                payload={"equity_usdt": equity_usdt, "base_margin_usdt": base_margin_usdt, "stop_dist_pct": float(settings.hard_stop_loss_pct), "ai_score": ai_score},
                            )

                        log_action(logger, "RISK_BUDGET_ADJUST", trace_id=trace_id, symbol=symbol,
                                   reason_code=ReasonCode.RISK_BUDGET_ADJUST_LEVERAGE.value, reason=risk_note,
                                   ai_score=ai_score, leverage_before=lev, leverage_after=lev2,
                                   stop_dist_pct=stop_dist_pct, client_order_id=client_order_id)

                        if hasattr(ex, "set_leverage_and_margin_mode"):
                            ex.set_leverage_and_margin_mode(symbol=symbol, leverage=lev)

                        client_order_id = make_client_order_id(
                            "sell",
                            symbol,
                            interval_minutes=settings.interval_minutes,
                            kline_open_time_ms=int(latest["open_time_ms"]),
                            trace_id=trace_id,
                        )
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                            side=Side.SELL.value, qty=qty, price=None, status="CREATED",
                            reason_code=ReasonCode.STRATEGY_SIGNAL,
                            reason="Setup B SELL",
                            payload={"latest": latest, "robot_score": score, "leverage": lev}
                        )
                        res = ex.place_market_order(symbol=symbol, side="SELL", qty=qty, client_order_id=client_order_id)
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                            event_type=_event_type_from_status(res.status),
                            side=Side.SELL.value, qty=qty, price=res.avg_price, status=res.status,
                            reason_code=open_reason_code, reason=open_reason, payload={"exchange": res.raw or {}, "open_reason": open_reason, "open_reason_code": open_reason_code.value}
                        )
                        meta2 = _parse_json_maybe(pos.get("meta_json") if pos else None)
                        trade_id2 = _find_open_trade_id(db, symbol, meta2)
                        save_position(db, symbol, 0.0, None, {"trace_id": trace_id, "note": "exited", "trade_id": trade_id2, "robot_score": score, "leverage": lev})
                        close_code = ReasonCode.STRATEGY_EXIT.value
                        if settings.take_profit_reason_on_positive_pnl and (res.pnl_usdt is not None and float(res.pnl_usdt) > 0):
                            close_code = ReasonCode.TAKE_PROFIT.value
                        if trade_id2 > 0:
                            _close_trade_and_train(
                                db,
                                settings,
                                metrics,
                                _load_ai_model(db, settings) if settings.ai_enabled else None,
                                trade_id=trade_id2,
                                symbol=symbol,
                                qty=float(qty),
                                exit_price=res.avg_price,
                                pnl_usdt=res.pnl_usdt,
                                close_reason_code=close_code,
                                close_reason="Setup B SELL",
                                trace_id=trace_id,
                            )
                        open_cnt = max(0, open_cnt - 1)

                        summary_kv = build_trade_summary(
                            event="SELL_FILLED",
                            trace_id=trace_id,
                            exchange=exchange,
                            symbol=symbol,
                            side="SELL",
                            qty=qty,
                            price=res.avg_price,
                            leverage=lev,
                            reason_code=close_code,
                            reason="Setup B SELL",
                            client_order_id=client_order_id,
                            exchange_order_id=res.exchange_order_id,
                            extra={"robot_score": round(float(score), 2), "fee_usdt": res.fee_usdt, "pnl_usdt": res.pnl_usdt},
                        )
                        send_trade_alert(
                            telegram,
                            title="平仓成交",
                            summary_kv=summary_kv,
                            payload={
                                "client_order_id": client_order_id,
                                "exchange_order_id": res.exchange_order_id,
                                "avg_price": res.avg_price,
                                "fee_usdt": res.fee_usdt,
                                "pnl_usdt": res.pnl_usdt,
                                "robot_score": score,
                                "leverage": lev,
                                "raw": res.raw or {},
                            },
                        )
                        log_action(
                            logger,
                            "SELL_FILLED",
                            trace_id=trace_id,
                            exchange=exchange,
                            symbol=symbol,
                            side="SELL",
                            qty=qty,
                            price=res.avg_price,
                            leverage=lev,
                            client_order_id=client_order_id,
                            exchange_order_id=res.exchange_order_id,
                            reason_code=close_code,
                            reason="Setup B SELL",
                            pnl_usdt=res.pnl_usdt,
                        )
                        metrics.orders_total.labels(SERVICE, settings.exchange, symbol, "SELL").inc()

                    metrics.last_tick_success.labels(SERVICE, symbol).set(1)

            # 如果触发紧急退出：在本轮处理完所有 symbol 之后清掉开关
            if get_flag(db, "EMERGENCY_EXIT", "false") == "true":
                set_flag(db, "EMERGENCY_EXIT", "false")

            # status snapshot for /admin/status
            try:
                upsert_service_status(
                    db,
                    service_name=SERVICE,
                    instance_id=instance_id,
                    status={
                        "trace_id": trace_id,
                        "last_tick_id": tick_id,
                        "last_tick_ts_utc": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
                        "last_tick_ts_hk": datetime.datetime.now(HK).isoformat(),
                        "symbols": getattr(settings, "symbols", []),
                        "open_positions": open_cnt if "open_cnt" in locals() else None,
                        "halt": get_flag(db, "HALT_TRADING", "false"),
                        "emergency": get_flag(db, "EMERGENCY_EXIT", "false"),
                    },
                )
            except Exception:
                pass

        except RateLimitError as e:
            sleep_s = e.retry_after_seconds or 2.0
            # 仅对 severe 的限流发送 Telegram，避免刷屏
            try:
                if getattr(e, "severe", False):
                    send_system_alert(
                        telegram,
                        title="⏳ 限流退避",
                        summary_kv=build_system_summary(
                            event="RATE_LIMIT_BACKOFF",
                            trace_id=trace_id,
                            exchange=exchange,
                            level="WARN",
                            reason_code=ReasonCode.RATE_LIMIT_429,
                            reason=f"rate limit backoff {float(sleep_s):.2f}s",
                            extra={
                                "group": getattr(e, "group", None),
                                "sleep_s": round(float(sleep_s), 2),
                                "severe": bool(getattr(e, "severe", False)),
                            },
                        ),
                        payload={
                            "group": getattr(e, "group", None),
                            "sleep_s": float(sleep_s),
                            "severe": bool(getattr(e, "severe", False)),
                        },
                    )
                log_action(
                    logger,
                    action="RATE_LIMIT_BACKOFF",
                    trace_id=trace_id,
                    reason_code=ReasonCode.RATE_LIMIT_429,
                    reason=f"rate limit backoff {float(sleep_s):.2f}s",
                    client_order_id=None,
                    extra={
                        "group": getattr(e, "group", None),
                        "sleep_s": float(sleep_s),
                        "severe": bool(getattr(e, "severe", False)),
                    },
                )
            except Exception:
                pass
            time.sleep(max(0.5, float(sleep_s)))
            continue
        except Exception as e:
            # 全局异常：避免某一个 symbol 的错误把整个服务打崩
            for sym in symbols:
                metrics.last_tick_success.labels(SERVICE, sym).set(0)
            try:
                send_system_alert(
                    telegram,
                    title="❌ 策略引擎异常",
                    summary_kv=build_system_summary(event="ENGINE_ERROR", trace_id=trace_id, exchange=exchange, level="ERROR", reason=str(e)[:200]),
                    payload={"error": str(e)},
                )
                log_action(logger, action="ENGINE_ERROR", trace_id=trace_id, reason_code="ERROR", reason=str(e)[:200], client_order_id=None)
            except Exception:
                pass

if __name__ == "__main__":
    main()