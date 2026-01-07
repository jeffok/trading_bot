from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.exchange import make_exchange
from shared.logging import get_logger, new_trace_id
from shared.redis import LeaderElector, redis_client
from shared.domain.heartbeat import upsert_service_status
from shared.domain.instance import get_instance_id
from shared.telemetry import Metrics, Telegram, start_metrics_http_server
from shared.domain.time import now_ms, HK

SERVICE = "data-syncer"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))


# ----------------------------
# Indicators (Setup-B friendly)
# ----------------------------

def _ema_update(prev: Optional[float], price: float, period: int) -> float:
    if prev is None:
        return price
    k = 2.0 / (period + 1.0)
    return price * k + prev * (1.0 - k)

def _rsi_update(
    closes: Deque[float],
    gains: Deque[float],
    losses: Deque[float],
    period: int,
    new_close: float,
) -> Optional[float]:
    """Streaming RSI using last diffs window (simple average version)."""
    if closes:
        diff = new_close - closes[-1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
        if len(gains) > period:
            gains.popleft()
            losses.popleft()
    closes.append(new_close)
    if len(closes) < period + 1:
        return None
    avg_gain = sum(gains) / period if period else 0.0
    avg_loss = sum(losses) / period if period else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _sma(values: Deque[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0

def _std(values: Deque[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _sma(values)
    var = sum((x - m) ** 2 for x in values) / float(len(values))
    return var ** 0.5


def compute_features_for_bars(
    bars: List[Dict[str, Any]],
    *,
    ema_fast_period: int = 7,
    ema_slow_period: int = 25,
    rsi_period: int = 14,
    atr_period: int = 14,
    adx_period: int = 14,
    bb_period: int = 20,
    mom_period: int = 10,
    vol_period: int = 20,
) -> List[Tuple[int, float, float, Optional[float], Dict[str, Any]]]:
    """Compute indicators for bars (ascending open_time_ms).

    Returns list of:
      (open_time_ms, ema_fast, ema_slow, rsi, features_dict)
    where features_dict is stored into features_json.
    """
    if not bars:
        return []

    # Streaming windows
    closes: Deque[float] = deque(maxlen=5000)
    gains: Deque[float] = deque(maxlen=rsi_period)
    losses: Deque[float] = deque(maxlen=rsi_period)

    bb_window: Deque[float] = deque(maxlen=bb_period)
    vol_window: Deque[float] = deque(maxlen=vol_period)
    ret_window: Deque[float] = deque(maxlen=vol_period)

    vol_sma_window: Deque[float] = deque(maxlen=bb_period)

    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None

    # ATR / ADX state (Wilder)
    prev_close: Optional[float] = None
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None

    tr_list: Deque[float] = deque(maxlen=atr_period)
    plus_dm_list: Deque[float] = deque(maxlen=atr_period)
    minus_dm_list: Deque[float] = deque(maxlen=atr_period)

    atr: Optional[float] = None
    plus_dm_s: Optional[float] = None
    minus_dm_s: Optional[float] = None

    dx_list: Deque[float] = deque(maxlen=adx_period)
    adx: Optional[float] = None

    out: List[Tuple[int, float, float, Optional[float], Dict[str, Any]]] = []

    for i, b in enumerate(bars):
        close = float(b["close_price"])
        high = float(b["high_price"])
        low = float(b["low_price"])
        volume = float(b["volume"])

        ema_fast = _ema_update(ema_fast, close, ema_fast_period)
        ema_slow = _ema_update(ema_slow, close, ema_slow_period)
        rsi = _rsi_update(closes, gains, losses, rsi_period, close)

        # Returns and momentum
        ret1 = None
        if prev_close is not None and prev_close != 0:
            ret1 = (close / prev_close) - 1.0
            ret_window.append(ret1)

        # Momentum (close - close_n)
        mom = None
        if len(closes) > mom_period:
            # closes includes new close at end
            mom = close - list(closes)[-1 - mom_period]

        # Bollinger
        bb_window.append(close)
        bb_mid = None
        bb_upper = None
        bb_lower = None
        bb_width = None
        if len(bb_window) == bb_period:
            mid = _sma(bb_window)
            sd = _std(bb_window)
            bb_mid = mid
            bb_upper = mid + 2.0 * sd
            bb_lower = mid - 2.0 * sd
            if mid != 0:
                bb_width = (bb_upper - bb_lower) / mid

        # Volume ratio
        vol_sma_window.append(volume)
        vol_sma = None
        vol_ratio = None
        if len(vol_sma_window) == vol_sma_window.maxlen:
            vol_sma = _sma(vol_sma_window)
            if vol_sma and vol_sma != 0:
                vol_ratio = volume / vol_sma

        # ATR / ADX
        atr14 = None
        adx14 = None
        plus_di = None
        minus_di = None

        if prev_close is not None and prev_high is not None and prev_low is not None:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            up_move = high - prev_high
            down_move = prev_low - low
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

            tr_list.append(tr)
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

            # Initial Wilder smoothing
            if atr is None and len(tr_list) == atr_period:
                atr = sum(tr_list) / float(atr_period)
                plus_dm_s = sum(plus_dm_list) / float(atr_period)
                minus_dm_s = sum(minus_dm_list) / float(atr_period)
            elif atr is not None:
                # Wilder smoothing update
                atr = atr - (atr / atr_period) + tr
                plus_dm_s = (plus_dm_s or 0.0) - ((plus_dm_s or 0.0) / atr_period) + plus_dm
                minus_dm_s = (minus_dm_s or 0.0) - ((minus_dm_s or 0.0) / atr_period) + minus_dm

            if atr is not None and atr != 0:
                plus_di = 100.0 * (float(plus_dm_s or 0.0) / atr)
                minus_di = 100.0 * (float(minus_dm_s or 0.0) / atr)
                denom = plus_di + minus_di
                if denom != 0:
                    dx = 100.0 * abs(plus_di - minus_di) / denom
                    dx_list.append(dx)

                    if adx is None and len(dx_list) == adx_period:
                        adx = sum(dx_list) / float(adx_period)
                    elif adx is not None:
                        adx = ((adx * (adx_period - 1)) + dx) / float(adx_period)

        if atr is not None:
            atr14 = float(atr)
        if adx is not None:
            adx14 = float(adx)

        # Volatility (std of returns)
        ret_std = None
        if len(ret_window) >= 2:
            ret_std = _std(ret_window)

        features: Dict[str, Any] = {
            "atr14": atr14,
            "adx14": adx14,
            "plus_di14": plus_di,
            "minus_di14": minus_di,
            "bb_mid20": bb_mid,
            "bb_upper20": bb_upper,
            "bb_lower20": bb_lower,
            "bb_width20": bb_width,
            "vol_sma20": vol_sma,
            "vol_ratio": vol_ratio,
            "mom10": mom,
            "ret1": ret1,
            "ret_std20": ret_std,
        }

        out.append((int(b["open_time_ms"]), float(ema_fast), float(ema_slow), rsi, features))

        prev_close = close
        prev_high = high
        prev_low = low

    return out


# ----------------------------
# DB helpers
# ----------------------------

def upsert_heartbeat(db: MariaDB, instance_id: str, status: dict):
    with db.tx() as cur:
        cur.execute(
            """
            INSERT INTO service_status (service_name, instance_id, last_heartbeat, status_json)
            VALUES (%s, %s, CURRENT_TIMESTAMP, %s)
            ON DUPLICATE KEY UPDATE last_heartbeat=CURRENT_TIMESTAMP, status_json=VALUES(status_json)
            """,
            (SERVICE, instance_id, json.dumps(status, ensure_ascii=False)),
        )

def _utc_now() -> datetime:
    return datetime.utcnow().replace(tzinfo=None)

def _hk_now() -> datetime:
    return datetime.now(HK)


def _archive_table_timestamp(
    db: MariaDB,
    *,
    src: str,
    dst: str,
    cutoff_days: int,
    trace_id: str,
    columns: str,
) -> int:
    """Archive rows from {src} to {dst} where created_at < now-interval, then delete from src.

    Note: history tables usually have an extra `archived_at` column with DEFAULT CURRENT_TIMESTAMP.
    We therefore MUST specify a column list to keep INSERT/SELECT column counts aligned.
    """
    cutoff = _utc_now() - timedelta(days=int(cutoff_days))
    with db.tx() as cur:
        cur.execute(
            f"INSERT IGNORE INTO {dst} ({columns}) SELECT {columns} FROM {src} WHERE created_at < %s",
            (cutoff,),
        )
        moved = cur.rowcount or 0
        cur.execute(
            f"DELETE FROM {src} WHERE created_at < %s",
            (cutoff,),
        )
        return int(moved)

def run_daily_archive(db: MariaDB, settings: Settings, metrics: Metrics, *, instance_id: str):
    """Run daily archive around HK midnight (00:00–00:05). Idempotent via system_config."""
    hk = _hk_now()
    if not (hk.hour == 0 and hk.minute <= 5):
        return

    hk_date = hk.strftime("%Y-%m-%d")
    key = "ARCHIVE_LAST_HK_DATE"
    last = db.fetch_one("SELECT value FROM system_config WHERE `key`=%s", (key,))
    if last and last["value"] == hk_date:
        return

    trace_id = new_trace_id("archive")
    metrics.archive_runs_total.labels(SERVICE).inc()

    moved_total = 0
    try:
        for src, dst, cols in [
            ("market_data", "market_data_history", "symbol,interval_minutes,open_time_ms,close_time_ms,open_price,high_price,low_price,close_price,volume,created_at"),
            ("market_data_cache", "market_data_cache_history", "symbol,interval_minutes,open_time_ms,ema_fast,ema_slow,rsi,features_json,created_at"),
            ("order_events", "order_events_history", "id,created_at,trace_id,service,exchange,symbol,client_order_id,exchange_order_id,event_type,side,qty,price,status,reason_code,reason,payload_json"),
            ("trade_logs", "trade_logs_history", "id,created_at,trace_id,actor,symbol,side,qty,leverage,stop_dist_pct,stop_price,client_order_id,exchange_order_id,robot_score,ai_prob,open_reason_code,open_reason,close_reason_code,close_reason,entry_time_ms,exit_time_ms,entry_price,exit_price,pnl,features_json,label,status"),
            ("position_snapshots", "position_snapshots_history", "id,created_at,symbol,base_qty,avg_entry_price,meta_json"),
        ]:
            moved = _archive_table_timestamp(db, src=src, dst=dst, cutoff_days=90, trace_id=trace_id, columns=cols)
            moved_total += moved

        with db.tx() as cur:
            cur.execute(
                """
                INSERT INTO archive_audit (trace_id, table_name, cutoff_days, moved_rows, message)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (trace_id, "ALL", 90, moved_total, f"archive done hk_date={hk_date}"),
            )
            cur.execute(
                """
                INSERT INTO system_config (`key`, `value`) VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE value=VALUES(value), updated_at=CURRENT_TIMESTAMP
                """,
                (key, hk_date),
            )

        metrics.archive_rows_total.labels(SERVICE).inc(moved_total)
        logger.info(f"archive_done trace_id={trace_id} moved_total={moved_total} hk_date={hk_date}")
    except Exception as e:
        metrics.archive_errors_total.labels(SERVICE).inc()
        logger.exception(f"archive_error trace_id={trace_id} err={e}")

# ----------------------------
# Precompute queue
# ----------------------------

def enqueue_precompute_tasks(
    db: MariaDB,
    *,
    symbol: str,
    interval_minutes: int,
    open_times: List[int],
    trace_id: str,
) -> int:
    if not open_times:
        return 0
    rows = [(symbol, interval_minutes, int(ot), "PENDING", 0, None, trace_id) for ot in open_times]
    with db.tx() as cur:
        cur.executemany(
            """
            INSERT IGNORE INTO precompute_tasks
              (symbol, interval_minutes, open_time_ms, status, try_count, last_error, trace_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        return cur.rowcount or 0

def _mark_tasks_done(db: MariaDB, *, symbol: str, interval_minutes: int, up_to_open_time_ms: int):
    with db.tx() as cur:
        cur.execute(
            """
            UPDATE precompute_tasks
            SET status='DONE'
            WHERE symbol=%s AND interval_minutes=%s AND status='PENDING' AND open_time_ms <= %s
            """,
            (symbol, interval_minutes, int(up_to_open_time_ms)),
        )

def _mark_tasks_error(db: MariaDB, *, symbol: str, interval_minutes: int, open_times: List[int], trace_id: str, err: str):
    if not open_times:
        return
    with db.tx() as cur:
        for ot in open_times:
            cur.execute(
                """
                UPDATE precompute_tasks
                SET status='ERROR', try_count=try_count+1, last_error=%s, trace_id=%s
                WHERE symbol=%s AND interval_minutes=%s AND open_time_ms=%s
                """,
                (err[:2000], trace_id, symbol, interval_minutes, int(ot)),
            )


def process_precompute_tasks(
    db: MariaDB,
    settings: Settings,
    metrics: Metrics,
    *,
    symbol: str,
    max_tasks: int = 800,
) -> int:
    """Process pending precompute tasks for one symbol; computes cache rows and marks tasks done."""
    interval = int(settings.interval_minutes)
    tasks = db.fetch_all(
        """
        SELECT open_time_ms FROM precompute_tasks
        WHERE symbol=%s AND interval_minutes=%s AND status='PENDING'
        ORDER BY open_time_ms ASC
        LIMIT %s
        """,
        (symbol, interval, int(max_tasks)),
    )
    if not tasks:
        return 0

    open_times = [int(r["open_time_ms"]) for r in tasks]
    min_ot = min(open_times)
    max_ot = max(open_times)

    interval_ms = interval * 60_000
    warmup_bars = 300
    warmup_start = max(0, min_ot - warmup_bars * interval_ms)

    # Fetch bars to compute
    bars = db.fetch_all(
        """
        SELECT open_time_ms, open_price, high_price, low_price, close_price, volume
        FROM market_data
        WHERE symbol=%s AND interval_minutes=%s AND open_time_ms >= %s AND open_time_ms <= %s
        ORDER BY open_time_ms ASC
        """,
        (symbol, interval, int(warmup_start), int(max_ot)),
    )
    if not bars:
        return 0

    # Compute features across warmup range, write only from min_ot onward
    t0 = time.time()
    computed = compute_features_for_bars(bars)

    cache_rows = []
    for ot, ema_f, ema_s, rsi, features in computed:
        if ot < min_ot:
            continue
        cache_rows.append(
            (
                symbol,
                interval,
                int(ot),
                ema_f,
                ema_s,
                rsi,
                json.dumps(features, ensure_ascii=False),
            )
        )

    if not cache_rows:
        return 0

    trace_id = new_trace_id("precompute")
    try:
        with db.tx() as cur:
            cur.executemany(
                """
                INSERT INTO market_data_cache
                  (symbol, interval_minutes, open_time_ms, ema_fast, ema_slow, rsi, features_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  ema_fast=VALUES(ema_fast),
                  ema_slow=VALUES(ema_slow),
                  rsi=VALUES(rsi),
                  features_json=VALUES(features_json)
                """,
                cache_rows,
            )
        _mark_tasks_done(db, symbol=symbol, interval_minutes=interval, up_to_open_time_ms=max_ot)

        metrics.precompute_tasks_processed_total.labels(SERVICE, symbol, str(interval)).inc(len(open_times))
        metrics.feature_compute_seconds.labels(SERVICE, symbol).observe(time.time() - t0)
        return len(open_times)
    except Exception as e:
        metrics.precompute_errors_total.labels(SERVICE, symbol, str(interval)).inc()
        _mark_tasks_error(db, symbol=symbol, interval_minutes=interval, open_times=open_times, trace_id=trace_id, err=str(e))
        logger.exception(f"precompute_error symbol={symbol} trace_id={trace_id} err={e}")
        return 0


# ----------------------------
# Sync + gap fill
# ----------------------------

def _insert_market_data(db: MariaDB, *, symbol: str, interval: int, klines) -> int:
    rows = [
        (symbol, interval, int(k.open_time_ms), int(k.close_time_ms), k.open, k.high, k.low, k.close, k.volume)
        for k in klines
    ]
    if not rows:
        return 0
    with db.tx() as cur:
        cur.executemany(
            """
            INSERT IGNORE INTO market_data
              (symbol, interval_minutes, open_time_ms, close_time_ms, open_price, high_price, low_price, close_price, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        return cur.rowcount or 0

def _fill_recent_gaps(db: MariaDB, ex, settings: Settings, metrics: Metrics, *, symbol: str, trace_id: str) -> int:
    """Detect gaps in last N bars and attempt to backfill missing klines."""
    interval = int(settings.interval_minutes)
    interval_ms = interval * 60_000
    recent = db.fetch_all(
        """
        SELECT open_time_ms FROM market_data
        WHERE symbol=%s AND interval_minutes=%s
        ORDER BY open_time_ms DESC LIMIT 600
        """,
        (symbol, interval),
    )
    if len(recent) < 3:
        return 0

    times = sorted([int(r["open_time_ms"]) for r in recent])
    missing_total = 0
    runs = 0
    for i in range(1, len(times)):
        gap = times[i] - times[i-1]
        if gap > interval_ms:
            runs += 1
            start = times[i-1] + interval_ms
            end = times[i] - interval_ms
            need = int((end - start) // interval_ms) + 1
            # fetch in chunks (limit 1000)
            cursor = start
            while cursor <= end:
                limit = min(1000, int((end - cursor) // interval_ms) + 1)
                kl = ex.fetch_klines(symbol=symbol, interval_minutes=interval, start_ms=int(cursor), limit=int(limit))
                inserted = _insert_market_data(db, symbol=symbol, interval=interval, klines=kl)
                if inserted > 0:
                    missing_total += inserted
                    # enqueue tasks for inserted open_times
                    open_times = [int(k.open_time_ms) for k in kl]
                    enq = enqueue_precompute_tasks(db, symbol=symbol, interval_minutes=interval, open_times=open_times, trace_id=trace_id)
                    metrics.precompute_tasks_enqueued_total.labels(SERVICE, symbol, str(interval)).inc(enq)
                # move cursor forward
                cursor = cursor + limit * interval_ms
    if runs:
        metrics.data_sync_gap_fill_runs_total.labels(SERVICE, symbol, str(interval)).inc(runs)
        metrics.data_sync_gap_fill_bars_total.labels(SERVICE, symbol, str(interval)).inc(missing_total)
    return missing_total


def sync_symbol_once(db: MariaDB, ex, settings: Settings, metrics: Metrics, telegram: Telegram, *, symbol: str, instance_id: str):
    interval = int(settings.interval_minutes)
    interval_ms = interval * 60_000
    trace_id = new_trace_id("sync")

    try:
        last = db.fetch_one(
            """
            SELECT open_time_ms FROM market_data
            WHERE symbol=%s AND interval_minutes=%s
            ORDER BY open_time_ms DESC LIMIT 1
            """,
            (symbol, interval),
        )
        start_ms = int(last["open_time_ms"]) + interval_ms if last else None

        klines = ex.fetch_klines(symbol=symbol, interval_minutes=interval, start_ms=start_ms, limit=1000)

        # gap detection within fetched batch (best-effort)
        if klines:
            for i in range(1, len(klines)):
                if (klines[i].open_time_ms - klines[i-1].open_time_ms) > interval_ms:
                    metrics.data_sync_gaps_total.labels(SERVICE, symbol, str(interval)).inc()
                    logger.warning(f"gap_detected symbol={symbol} interval={interval} prev={klines[i-1].open_time_ms} cur={klines[i].open_time_ms}")

        if not klines:
            upsert_heartbeat(db, instance_id, {"trace_id": trace_id, "status": "NO_DATA", "symbol": symbol})
            return

        inserted = _insert_market_data(db, symbol=symbol, interval=interval, klines=klines)
        if inserted:
            open_times = [int(k.open_time_ms) for k in klines]
            enq = enqueue_precompute_tasks(db, symbol=symbol, interval_minutes=interval, open_times=open_times, trace_id=trace_id)
            metrics.precompute_tasks_enqueued_total.labels(SERVICE, symbol, str(interval)).inc(enq)

        # Compute lag based on cache
        last_cache = db.fetch_one(
            """
            SELECT open_time_ms FROM market_data_cache
            WHERE symbol=%s AND interval_minutes=%s
            ORDER BY open_time_ms DESC LIMIT 1
            """,
            (symbol, interval),
        )
        if last_cache:
            lag = int(now_ms() - int(last_cache["open_time_ms"]))
            metrics.data_sync_lag_ms.labels(SERVICE, symbol, str(interval)).set(lag)

        # gap fill on recent history
        _fill_recent_gaps(db, ex, settings, metrics, symbol=symbol, trace_id=trace_id)

        upsert_heartbeat(
            db,
            instance_id,
            {"trace_id": trace_id, "status": "OK", "symbol": symbol, "inserted": inserted},
        )
    except Exception as e:
        metrics.data_sync_errors_total.labels(SERVICE).inc()
        upsert_heartbeat(db, instance_id, {"trace_id": trace_id, "status": "ERROR", "symbol": symbol, "error": str(e)})
        logger.exception(f"sync_error symbol={symbol} trace_id={trace_id} err={e}")
        telegram.send(f"[{SERVICE}] sync_error symbol={symbol} trace_id={trace_id} err={e}")


def main():
    settings = Settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, migrations_dir=str(Path(__file__).resolve().parents[2] / "migrations"))

    metrics = Metrics(SERVICE)
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    # Expose metrics
    port = settings.metrics_port if settings.metrics_port else 9101
    start_metrics_http_server(port)
    logger.info(f"metrics_http_server_started port={port}")

    ex = make_exchange(settings, metrics=metrics, service_name=SERVICE)

    instance_id = get_instance_id(settings.instance_id)

    r = redis_client(settings.redis_url)
    leader_key = f"{settings.leader_key_prefix}:{SERVICE}"
    elector = LeaderElector(
        r,
        key=leader_key,
        instance_id=instance_id,
        ttl_seconds=settings.leader_ttl_seconds,
        renew_interval_seconds=settings.leader_renew_interval_seconds,
    )
    last_role: str = "unknown"

    symbols = list(settings.symbols) if getattr(settings, "symbols", None) else [settings.symbol]
    # safety: de-dupe
    seen = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]
    logger.info(f"start service={SERVICE} exchange={settings.exchange} interval={settings.interval_minutes} symbols={symbols}")

    while True:
        # leader election: only leader performs sync/archive/precompute
        is_leader = True
        if settings.leader_election_enabled:
            is_leader = elector.ensure()
        metrics.leader_is_leader.labels(SERVICE, instance_id).set(1 if is_leader else 0)

        role = "leader" if is_leader else "follower"
        if role != last_role:
            metrics.leader_changes_total.labels(SERVICE, instance_id, role).inc()
            last_role = role

        # heartbeat
        try:
            upsert_service_status(
                db,
                service_name=SERVICE,
                instance_id=instance_id,
                trace_id=new_trace_id("hb"),
                status=role,
                extra={"leader": elector.get_leader() if settings.leader_election_enabled else instance_id},
            )
        except Exception:
            pass

        if not is_leader:
            time.sleep(settings.leader_follower_sleep_seconds)
            continue

        metrics.data_sync_cycles_total.labels(SERVICE).inc()
        # daily archive
        run_daily_archive(db, settings, metrics, instance_id=instance_id)

        for sym in symbols:
            sync_symbol_once(db, ex, settings, metrics, telegram, symbol=sym, instance_id=instance_id)
            # process a slice of precompute tasks per symbol each loop
            processed = process_precompute_tasks(db, settings, metrics, symbol=sym, max_tasks=800)
            if processed:
                logger.info(f"precompute_done symbol={sym} processed={processed}")

        time.sleep(10)


if __name__ == "__main__":
    main()
