
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.exchange import make_exchange
from shared.ai import OnlineLogisticRegression
from shared.domain.system_config import get_system_config, write_system_config
from shared.logging import get_logger, new_trace_id
from shared.redis import distributed_lock, redis_client
from shared.telemetry import Metrics, Telegram, start_metrics_http_server
from shared.domain.enums import OrderEventType, ReasonCode, Side
from shared.domain.heartbeat import upsert_service_status
from shared.domain.instance import get_instance_id
from shared.domain.events import append_order_event
from shared.domain.idempotency import make_client_order_id
from shared.domain.time import next_tick_sleep_seconds

SERVICE = "strategy-engine"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))

def _normalize_status(s: str) -> str:
    return str(s or "").upper()


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
        WHERE oe.event_type IN ('CREATED','SUBMITTED')
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
            if status_u in ("FILLED", "CLOSED"):
                terminal_event = OrderEventType.FILLED
            elif status_u in ("CANCELED", "CANCELLED"):
                terminal_event = OrderEventType.CANCELED
            elif status_u in ("REJECTED", "EXPIRED", "ERROR", "FAILED"):
                terminal_event = OrderEventType.ERROR

            payload = getattr(st, "raw", None) or {}
            qty = float(getattr(st, "filled_qty", 0.0) or 0.0)
            avg_price = getattr(st, "avg_price", None)

            trace_id = new_trace_id("reconcile")

            if terminal_event is not None:
                append_order_event(
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
            telegram.send(f"[{SERVICE}] RECONCILE ERROR symbol={symbol} client_order_id={client_order_id} err={e}")
            continue

    return fixed

def get_flag(db: MariaDB, key: str, default: str = "false") -> str:
    row = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
    return (row["value"] if row else default).strip().lower()

def set_flag(db: MariaDB, key: str, value: str) -> None:
    db.execute("INSERT INTO system_config(`key`,`value`) VALUES (%s,%s) ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)", (key, value))

def latest_cache(db: MariaDB, symbol: str, interval_minutes: int):
    return db.fetch_one(
        """
        SELECT m.open_time_ms, m.close_price, c.ema_fast, c.ema_slow, c.rsi, c.features_json
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


def _load_ai_model(db: MariaDB, settings: Settings) -> OnlineLogisticRegression:
    dim = 12
    raw = get_system_config(db, settings.ai_model_key, default="")
    if raw:
        try:
            return OnlineLogisticRegression.from_dict(json.loads(raw), fallback_dim=dim)
        except Exception:
            pass


    return OnlineLogisticRegression(dim=dim, lr=float(settings.ai_lr), l2=float(settings.ai_l2))


def _maybe_persist_ai_model(db: MariaDB, settings: Settings, model: OnlineLogisticRegression, *, trace_id: str, force: bool = False) -> None:
    # Persist every 10 updates or on force.
    if not force and (int(model.seen) % 10) != 0:
        return
    try:
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
    model: OnlineLogisticRegression | None,
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
    settings = Settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, Path("/app/migrations"))

    metrics = Metrics(SERVICE)
    start_metrics_http_server(int(settings.metrics_port) or 9102)
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
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
            upsert_service_status(db, service_name=SERVICE, instance_id=instance_id, status={"status": "RUNNING", "role": role, "leader": elector.get_leader() if settings.leader_election_enabled else instance_id})
        except Exception:
            pass

        if not is_leader:
            time.sleep(settings.leader_follower_sleep_seconds)
            continue

        time.sleep(next_tick_sleep_seconds(settings.strategy_tick_seconds))
        trace_id = new_trace_id("tick")

        try:
            if get_flag(db, "HALT_TRADING", "false") == "true":
                telegram.send(f"[HALT] 本轮跳过 trace_id={trace_id} symbols={','.join(settings.symbols)}")
                continue

            tick_id = int(time.time() // settings.strategy_tick_seconds)

            # best-effort reconcile (stale CREATED/SUBMITTED orders)
            try:
                fixed = reconcile_stale_orders(db, ex, exchange_name=settings.exchange, max_age_seconds=180, metrics=metrics, telegram=telegram)
                if fixed:
                    logger.info(f"reconcile_fixed={fixed} trace_id={trace_id}")
            except Exception:
                pass

            # 先计算“当前全局已持仓数量”，用于限制最多 3 单（跨交易对）
            pos_map = get_latest_positions_map(db, settings.symbols)
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
                    for s in settings.symbols:
                        if float(pos_map.get(s, 0.0) or 0.0) > 0.0:
                            continue  # 已有仓位的币对不参与“选币开仓”，但仍会参与后续平仓/止损逻辑
                        latest_s = latest_cache(db, s, settings.interval_minutes)
                        if not latest_s:
                            continue
                        sig_s = setup_b_signal(latest_s)
                        if sig_s != "BUY":
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
                        meta = {"robot_score": float(score_s), "ai_prob": ai_prob, "combined_score": combined, "features_bundle": feat_bundle}
                        candidates.append((float(combined), s, meta))
                    # 按评分从高到低选择前 N 个开仓候选
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    selected_open_symbols = set([sym for _, sym, _ in candidates[:available_slots]])
                    selected_open_meta = {sym: meta for _, sym, meta in candidates[:available_slots]}
            except Exception:
                # 选币失败不应导致主循环崩溃：回退为“无候选”，本轮不主动开仓
                selected_open_symbols = set()
                selected_open_meta = {}
            for symbol in settings.symbols:
                lock_key = f"lock:tick:{settings.exchange}:{symbol}:{tick_id}"
                with distributed_lock(r, lock_key, ttl_ms=int(settings.strategy_tick_seconds * 1000 * 0.9)) as acquired:
                    if not acquired:
                        continue

                    latest = latest_cache(db, symbol, settings.interval_minutes)
                    if not latest:
                        continue

                    last_price = float(latest["close_price"])
                    if hasattr(ex, "update_last_price"):
                        ex.update_last_price(symbol, last_price)

                    pos = get_position(db, symbol)
                    base_qty = float(pos["base_qty"]) if pos else 0.0
                    avg_entry = float(pos["avg_entry_price"]) if pos and pos["avg_entry_price"] is not None else None

                    # --- 紧急退出：对所有交易对生效 ---
                    if get_flag(db, "EMERGENCY_EXIT", "false") == "true":
                        if base_qty > 0:
                            client_order_id = make_client_order_id(
                                "exit",
                                symbol,
                                kline_open_time_ms=int(latest["open_time_ms"]),
                            )
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                                side=Side.SELL.value, qty=base_qty, price=None, status="CREATED",
                                reason_code=ReasonCode.EMERGENCY_EXIT, reason="Emergency exit requested", payload={}
                            )
                            res = ex.place_market_order(symbol=symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                                event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
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
                            telegram.send_alert_zh(
                                title="紧急退出已执行",
                                summary_kv={
                                    "event": "EMERGENCY_EXIT_EXECUTED",
                                    "trace_id": trace_id,
                                    "exchange": settings.exchange,
                                    "symbol": symbol,
                                    "side": "SELL",
                                    "qty": base_qty,
                                    "price": res.avg_price,
                                    "fee_usdt": res.fee_usdt,
                                    "pnl_usdt": res.pnl_usdt,
                                },
                                payload={
                                    "client_order_id": client_order_id,
                                    "exchange_order_id": res.exchange_order_id,
                                    "avg_price": res.avg_price,
                                    "fee_usdt": res.fee_usdt,
                                    "pnl_usdt": res.pnl_usdt,
                                    "raw": res.raw or {},
                                },
                            )

                        # 当所有币对都检查完后再清掉 EMERGENCY_EXIT
                        continue

                    # --- 硬止损（逐仓合约） ---
                    if base_qty > 0 and avg_entry is not None:
                        meta = _parse_json_maybe(pos.get('meta_json') if pos else None)
                        stop_dist_pct = float(meta.get('stop_dist_pct') or settings.hard_stop_loss_pct)
                        stop_price = float(meta.get('stop_price') or (avg_entry * (1.0 - stop_dist_pct)))
                        if last_price <= stop_price:
                            client_order_id = make_client_order_id(
                                "sl",
                                symbol,
                                kline_open_time_ms=int(latest["open_time_ms"]),
                            )
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                                side=Side.SELL.value, qty=base_qty, price=None, status="CREATED",
                                reason_code=ReasonCode.STOP_LOSS,
                                reason=f"Hard stop loss: last={last_price} <= stop={stop_price}",
                                payload={"last_price": last_price, "stop_price": stop_price}
                            )
                            res = ex.place_market_order(symbol=symbol, side="SELL", qty=base_qty, client_order_id=client_order_id)
                            append_order_event(
                                db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                                client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                                event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
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
                            telegram.send_alert_zh(
                                title="触发止损",
                                summary_kv={
                                    "event": "STOP_LOSS",
                                    "trace_id": trace_id,
                                    "exchange": settings.exchange,
                                    "symbol": symbol,
                                    "side": "SELL",
                                    "qty": base_qty,
                                    "last_price": round(float(last_price), 4),
                                    "stop_price": round(float(stop_price), 4),
                                    "fee_usdt": res.fee_usdt,
                                    "pnl_usdt": res.pnl_usdt,
                                },
                                payload={
                                    "client_order_id": client_order_id,
                                    "exchange_order_id": res.exchange_order_id,
                                    "avg_price": res.avg_price,
                                    "fee_usdt": res.fee_usdt,
                                    "pnl_usdt": res.pnl_usdt,
                                    "raw": res.raw or {},
                                },
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
                            kline_open_time_ms=int(latest["open_time_ms"]),
                        )

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
                            open_reason_code=ReasonCode.AI_SELECT.value if ai_prob is not None else ReasonCode.STRATEGY_SIGNAL.value,
                            open_reason=open_reason,
                            features_bundle=feat_bundle if isinstance(feat_bundle, dict) else {},
                        )
                        metrics.trades_open_total.labels(SERVICE, symbol).inc()

                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                            side=Side.BUY.value, qty=qty, price=None, status="CREATED",
                            reason_code=ReasonCode.STRATEGY_SIGNAL,
                            reason="Setup B BUY",
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
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                            event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
                            side=Side.BUY.value, qty=qty, price=res.avg_price, status=res.status,
                            reason_code=ReasonCode.STRATEGY_SIGNAL, reason="Order placed", payload=res.raw or {}
                        )
                        entry_price = res.avg_price if res.avg_price is not None else last_price
                        save_position(db, symbol, qty, float(entry_price), {"trace_id": trace_id, "note": "entered", "robot_score": score, "leverage": lev})
                        open_cnt += 1

                        telegram.send_alert_zh(
                            title="开仓成交",
                            summary_kv={
                                "event": "BUY_FILLED",
                                "trace_id": trace_id,
                                "exchange": settings.exchange,
                                "symbol": symbol,
                                "side": "BUY",
                                "qty": qty,
                                "price": round(float(entry_price), 4),
                                "leverage": lev,
                                "robot_score": round(float(score), 2),
                                "min_margin_usdt": settings.min_order_usdt,
                                "notional_min_usdt": round(float(settings.min_order_usdt) * float(lev), 4),
                                "fee_usdt": res.fee_usdt,
                                "pnl_usdt": res.pnl_usdt,
                            },
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
                        metrics.orders_total.labels(SERVICE, settings.exchange, symbol, "BUY").inc()

                    elif sig == "SELL" and base_qty > 0:
                        qty = base_qty
                        score = compute_robot_score(latest, signal="SELL")
                        lev = leverage_from_score(settings, score)
                        if hasattr(ex, "set_leverage_and_margin_mode"):
                            ex.set_leverage_and_margin_mode(symbol=symbol, leverage=lev)

                        client_order_id = make_client_order_id(
                            "sell",
                            symbol,
                            kline_open_time_ms=int(latest["open_time_ms"]),
                        )
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=None, event_type=OrderEventType.CREATED,
                            side=Side.SELL.value, qty=qty, price=None, status="CREATED",
                            reason_code=ReasonCode.STRATEGY_SIGNAL,
                            reason="Setup B SELL",
                            payload={"latest": latest, "robot_score": score, "leverage": lev}
                        )
                        res = ex.place_market_order(symbol=symbol, side="SELL", qty=qty, client_order_id=client_order_id)
                        append_order_event(
                            db, trace_id=trace_id, service=SERVICE, exchange=settings.exchange, symbol=symbol,
                            client_order_id=client_order_id, exchange_order_id=res.exchange_order_id,
                            event_type=OrderEventType.FILLED if res.status.upper() == "FILLED" else OrderEventType.SUBMITTED,
                            side=Side.SELL.value, qty=qty, price=res.avg_price, status=res.status,
                            reason_code=ReasonCode.STRATEGY_SIGNAL, reason="Order placed", payload=res.raw or {}
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

                        telegram.send_alert_zh(
                            title="平仓成交",
                            summary_kv={
                                "event": "SELL_FILLED",
                                "trace_id": trace_id,
                                "exchange": settings.exchange,
                                "symbol": symbol,
                                "side": "SELL",
                                "qty": qty,
                                "price": res.avg_price,
                                "leverage": lev,
                                "robot_score": round(float(score), 2),
                                "fee_usdt": res.fee_usdt,
                                "pnl_usdt": res.pnl_usdt,
                            },
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
                        "symbols": getattr(settings, "symbols", []),
                        "open_positions": open_cnt if "open_cnt" in locals() else None,
                        "halt": get_flag(db, "HALT_TRADING", "false"),
                        "emergency": get_flag(db, "EMERGENCY_EXIT", "false"),
                    },
                )
            except Exception:
                pass

        except Exception as e:
            # 全局异常：避免某一个 symbol 的错误把整个服务打崩
            for sym in settings.symbols:
                metrics.last_tick_success.labels(SERVICE, sym).set(0)
            telegram.send(f"[{SERVICE}] ERROR trace_id={trace_id} err={e}")

if __name__ == "__main__":
    main()
