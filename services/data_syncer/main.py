from __future__ import annotations

import json
import os
import time
import statistics
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

from shared.config import Settings, load_settings
from shared.db import MariaDB, migrate
from shared.exchange import make_exchange
from shared.exchange.errors import RateLimitError
from shared.logging import get_logger, new_trace_id
from shared.redis import LeaderElector, redis_client
from shared.domain.heartbeat import upsert_service_status
from shared.domain.instance import get_instance_id
from shared.domain.runtime_config import RuntimeConfig
from shared.telemetry import Metrics, Telegram, start_metrics_http_server
from shared.telemetry.system_alerts import send_system_alert, build_system_summary
from shared.domain.time import now_ms, HK
from shared.domain.events import append_error_event

SERVICE = "data-syncer"
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))

# Per-symbol lag alert throttling (in-memory)
LAG_ALERT_LAST_AT: dict[str, float] = {}


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
    kc_period: int = 20,
    kc_mult: float = 1.5,
    mom_period: int = 10,
    vol_period: int = 20,
) -> List[Tuple[int, float, float, Optional[float], Dict[str, Any]]]:
    """Compute indicators for bars (ascending open_time_ms).

    Returns list of:
      (open_time_ms, ema_fast, ema_slow, rsi, features_dict)

    Notes (V8.3 / Setup B):
    - Adds Keltner Channel + Squeeze status + RSI slope (for Setup B).
    - BTC correlation is computed in process_precompute_tasks (needs BTC series).
    """

    if not bars:
        return []

    # helpers (keep minimal and stable; no heavy deps)
    def _ema_update(prev: Optional[float], price: float, period: int) -> float:
        a = 2.0 / (period + 1.0)
        return price if prev is None else (a * price + (1.0 - a) * prev)

    def _rsi_update(closes: Deque[float], gains: Deque[float], losses: Deque[float], period: int, close: float) -> Optional[float]:
        if closes:
            chg = close - closes[-1]
            gains.append(max(chg, 0.0))
            losses.append(max(-chg, 0.0))
        closes.append(close)
        # warmup
        if len(gains) < period or len(losses) < period:
            return None
        avg_gain = sum(list(gains)[-period:]) / period
        avg_loss = sum(list(losses)[-period:]) / period
        if avg_loss <= 1e-12:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
        if prev_close is None:
            return high - low
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    out: List[Tuple[int, float, float, Optional[float], Dict[str, Any]]] = []

    closes: Deque[float] = deque(maxlen=max(bb_period, mom_period, vol_period, rsi_period, 200) + 5)
    highs: Deque[float] = deque(maxlen=200)
    lows: Deque[float] = deque(maxlen=200)
    vols: Deque[float] = deque(maxlen=200)
    gains: Deque[float] = deque(maxlen=200)
    losses: Deque[float] = deque(maxlen=200)
    rsis: Deque[Optional[float]] = deque(maxlen=200)

    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    kc_mid: Optional[float] = None

    prev_close: Optional[float] = None

    # ATR / ADX state
    trs: Deque[float] = deque(maxlen=200)
    dm_plus: Deque[float] = deque(maxlen=200)
    dm_minus: Deque[float] = deque(maxlen=200)
    dxs: Deque[float] = deque(maxlen=200)

    prev_high: Optional[float] = None
    prev_low: Optional[float] = None

    def _sma(vals: List[float]) -> Optional[float]:
        return (sum(vals) / len(vals)) if vals else None

    def _std(vals: List[float]) -> Optional[float]:
        if len(vals) < 2:
            return None
        try:
            return float(statistics.pstdev(vals))
        except Exception:
            return None

    for i, b in enumerate(bars):
        ot = int(b["open_time_ms"])
        close = float(b["close_price"])
        high = float(b["high_price"])
        low = float(b["low_price"])
        volume = float(b["volume"])

        # base streams
        highs.append(high)
        lows.append(low)
        vols.append(volume)

        # EMA
        ema_fast = _ema_update(ema_fast, close, ema_fast_period)
        ema_slow = _ema_update(ema_slow, close, ema_slow_period)
        kc_mid = _ema_update(kc_mid, close, kc_period)

        # RSI
        rsi = _rsi_update(closes, gains, losses, rsi_period, close)
        rsis.append(rsi)

        # Returns and momentum
        ret1 = None
        if prev_close is not None and prev_close != 0:
            ret1 = (close / prev_close) - 1.0
        # momentum based on close
        mom = None
        if len(closes) > mom_period:
            prev_n = list(closes)[-mom_period - 1]
            if prev_n != 0:
                mom = (close / prev_n) - 1.0

        # vol ratio
        vol_sma = _sma(list(vols)[-vol_period:]) if len(vols) >= vol_period else None
        vol_ratio = (volume / vol_sma) if (vol_sma and vol_sma > 0) else None

        # ATR / DI / ADX
        tr = _true_range(high, low, prev_close)
        trs.append(tr)

        if prev_high is None or prev_low is None:
            dmp = 0.0
            dmn = 0.0
        else:
            up_move = high - prev_high
            down_move = prev_low - low
            dmp = up_move if (up_move > down_move and up_move > 0) else 0.0
            dmn = down_move if (down_move > up_move and down_move > 0) else 0.0
        dm_plus.append(dmp)
        dm_minus.append(dmn)

        atr14 = _sma(list(trs)[-atr_period:]) if len(trs) >= atr_period else None
        plus_di = None
        minus_di = None
        dx = None
        adx14 = None
        if atr14 and atr14 > 1e-12 and len(dm_plus) >= adx_period and len(dm_minus) >= adx_period:
            sum_tr = sum(list(trs)[-adx_period:])
            sum_p = sum(list(dm_plus)[-adx_period:])
            sum_m = sum(list(dm_minus)[-adx_period:])
            if sum_tr > 1e-12:
                plus_di = 100.0 * (sum_p / sum_tr)
                minus_di = 100.0 * (sum_m / sum_tr)
                denom = (plus_di + minus_di)
                if denom and denom > 1e-12:
                    dx = 100.0 * abs(plus_di - minus_di) / denom
                    dxs.append(float(dx))
                    adx14 = _sma(list(dxs)[-adx_period:]) if len(dxs) >= adx_period else None

        # Bollinger bands (SMA + std)
        bb_mid = None
        bb_upper = None
        bb_lower = None
        bb_width = None
        if len(closes) >= bb_period:
            window = list(closes)[-bb_period:]
            bb_mid = _sma(window)
            sd = _std(window)
            if bb_mid is not None and sd is not None:
                bb_upper = bb_mid + 2.0 * sd
                bb_lower = bb_mid - 2.0 * sd
                if bb_mid != 0:
                    bb_width = (bb_upper - bb_lower) / abs(bb_mid)

        # Keltner channel (EMA mid + ATR*mult)
        kc_upper = None
        kc_lower = None
        if kc_mid is not None and atr14 is not None:
            kc_upper = float(kc_mid) + float(kc_mult) * float(atr14)
            kc_lower = float(kc_mid) - float(kc_mult) * float(atr14)

        # Squeeze: Bollinger inside Keltner
        squeeze_status = None
        if bb_upper is not None and bb_lower is not None and kc_upper is not None and kc_lower is not None:
            squeeze_status = 1 if (bb_upper < kc_upper and bb_lower > kc_lower) else 0

        # RSI slope over 5 bars
        rsi_slope5 = None
        if rsi is not None and len(rsis) >= 6:
            rsi_5 = list(rsis)[-6]
            if rsi_5 is not None:
                rsi_slope5 = float(rsi) - float(rsi_5)

        # ret std
        ret_std = None
        if ret1 is not None and len(closes) >= vol_period + 1:
            rets = []
            cl = list(closes)[-(vol_period + 1):]
            for j in range(1, len(cl)):
                if cl[j-1] != 0:
                    rets.append((cl[j] / cl[j-1]) - 1.0)
            ret_std = _std(rets) if rets else None

        features: Dict[str, Any] = {
            "atr14": atr14,
            "adx14": adx14,
            "plus_di14": plus_di,
            "minus_di14": minus_di,
            "bb_mid20": bb_mid,
            "bb_upper20": bb_upper,
            "bb_lower20": bb_lower,
            "bb_width20": bb_width,
            "kc_mid20": kc_mid,
            "kc_upper20": kc_upper,
            "kc_lower20": kc_lower,
            "squeeze_status": squeeze_status,
            "vol_sma20": vol_sma,
            "vol_ratio": vol_ratio,
            "mom10": mom,
            "ret1": ret1,
            "ret_std20": ret_std,
            "rsi_slope5": rsi_slope5,
        }

        out.append((ot, float(ema_fast or 0.0), float(ema_slow or 0.0), rsi, features))

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
            ("market_data_cache", "market_data_cache_history", "symbol,interval_minutes,open_time_ms,feature_version,ema_fast,ema_slow,rsi,features_json,created_at"),
            ("order_events", "order_events_history", "id,created_at,trace_id,service,exchange,symbol,client_order_id,exchange_order_id,event_type,side,qty,price,status,reason_code,reason,payload_json"),
            ("trade_logs", "trade_logs_history", "id,created_at,trace_id,actor,symbol,side,qty,leverage,stop_dist_pct,stop_price,client_order_id,exchange_order_id,stop_client_order_id,stop_exchange_order_id,stop_order_type,robot_score,ai_prob,open_reason_code,open_reason,close_reason_code,close_reason,entry_time_ms,exit_time_ms,entry_price,exit_price,pnl,features_json,label,status,updated_at"),
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
    feature_version: int = 1,
) -> int:
    if not open_times:
        return 0
    fv = int(feature_version or 1)
    rows = [(symbol, interval_minutes, int(ot), fv, "PENDING", 0, None, trace_id) for ot in open_times]
    with db.tx() as cur:
        cur.executemany(
            """
            INSERT IGNORE INTO precompute_tasks
              (symbol, interval_minutes, open_time_ms, feature_version, status, try_count, last_error, trace_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        return cur.rowcount or 0

def _mark_tasks_done(db: MariaDB, *, symbol: str, interval_minutes: int, feature_version: int, up_to_open_time_ms: int):
    with db.tx() as cur:
        cur.execute(
            """
            UPDATE precompute_tasks
            SET status='DONE'
            WHERE symbol=%s AND interval_minutes=%s AND feature_version=%s AND status='PENDING' AND open_time_ms <= %s
            """,
            (symbol, interval_minutes, int(feature_version or 1), int(up_to_open_time_ms)),
        )

def _mark_tasks_error(db: MariaDB, *, symbol: str, interval_minutes: int, feature_version: int, open_times: List[int], trace_id: str, err: str):
    if not open_times:
        return
    with db.tx() as cur:
        for ot in open_times:
            cur.execute(
                """
                UPDATE precompute_tasks
                SET status='ERROR', try_count=try_count+1, last_error=%s, trace_id=%s
                WHERE symbol=%s AND interval_minutes=%s AND feature_version=%s AND open_time_ms=%s
                """,
                (err[:2000], trace_id, symbol, interval_minutes, int(feature_version or 1), int(ot)),
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
        WHERE symbol=%s AND interval_minutes=%s AND feature_version=%s AND status='PENDING'
        ORDER BY open_time_ms ASC
        LIMIT %s
        """,
        (symbol, interval, int(settings.feature_version), int(max_tasks)),
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
                int(settings.feature_version),
                ema_f,
                ema_s,
                rsi,
                json.dumps(features, ensure_ascii=False),
            )
        )


    # ---- V8.3: BTC correlation feature (best-effort) ----
    if symbol != getattr(settings, "btc_symbol", "BTCUSDT"):
        try:
            btc_symbol = getattr(settings, "btc_symbol", "BTCUSDT")
            ots = [int(r[2]) for r in cache_rows]  # (symbol, interval, ot, ...)
            if ots:
                min_cache_ot = int(min(ots))
                max_cache_ot = int(max(ots))
                btc_rows = db.fetch_all(
                    """
                    SELECT open_time_ms, close_price
                    FROM market_data
                    WHERE symbol=%s AND interval_minutes=%s AND open_time_ms BETWEEN %s AND %s
                    ORDER BY open_time_ms ASC
                    """,
                    (btc_symbol, interval, min_cache_ot, max_cache_ot),
                ) or []
                btc_close_by_ot = {
                    int(r["open_time_ms"]): float(r["close_price"])
                    for r in btc_rows
                    if r.get("open_time_ms") is not None and r.get("close_price") is not None
                }

                # compute btc ret1 per open_time_ms
                btc_ret_by_ot = {}
                prev_btc_close = None
                for ot in sorted(btc_close_by_ot.keys()):
                    c = btc_close_by_ot[ot]
                    if prev_btc_close is not None and prev_btc_close != 0:
                        btc_ret_by_ot[ot] = (c / prev_btc_close) - 1.0
                    else:
                        btc_ret_by_ot[ot] = None
                    prev_btc_close = c

                # build local series for symbol ret1 from existing rows
                sym_ret_by_ot = {}
                for r in cache_rows:
                    try:
                        f = json.loads(r[7] or "{}")
                    except Exception:
                        f = {}
                    sym_ret_by_ot[int(r[2])] = f.get("ret1")

                def _pearson(xs, ys):
                    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
                    if len(pairs) < 20:
                        return None
                    xvals = [p[0] for p in pairs]
                    yvals = [p[1] for p in pairs]
                    mx = sum(xvals) / len(xvals)
                    my = sum(yvals) / len(yvals)
                    num = sum((x - mx) * (y - my) for x, y in pairs)
                    denx = sum((x - mx) ** 2 for x in xvals)
                    deny = sum((y - my) ** 2 for y in yvals)
                    if denx <= 1e-18 or deny <= 1e-18:
                        return None
                    return float(num / ((denx ** 0.5) * (deny ** 0.5)))

                # rolling correlation window
                window = 96
                ots_sorted = sorted(sym_ret_by_ot.keys())
                corr_by_ot = {}
                for i2, ot in enumerate(ots_sorted):
                    start_i = max(0, i2 - window + 1)
                    w_ots = ots_sorted[start_i : i2 + 1]
                    xs = [sym_ret_by_ot.get(x) for x in w_ots]
                    ys = [btc_ret_by_ot.get(x) for x in w_ots]
                    corr_by_ot[ot] = _pearson(xs, ys)

                cache_rows2 = []
                for row in cache_rows:
                    ot = int(row[2])
                    try:
                        feats = json.loads(row[7] or "{}")
                    except Exception:
                        feats = {}
                    feats["btc_corr96"] = corr_by_ot.get(ot)
                    cache_rows2.append(
                        (row[0], row[1], row[2], row[3], row[4], row[5], row[6], json.dumps(feats, ensure_ascii=False))
                    )
                cache_rows = cache_rows2
        except Exception:
            # correlation is best-effort; ignore failures
            pass

    trace_id = new_trace_id("precompute")
    try:
        with db.tx() as cur:
            cur.executemany(
                """
                INSERT INTO market_data_cache
                  (symbol, interval_minutes, open_time_ms, feature_version, ema_fast, ema_slow, rsi, features_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  ema_fast=VALUES(ema_fast),
                  ema_slow=VALUES(ema_slow),
                  rsi=VALUES(rsi),
                  features_json=VALUES(features_json)
                """,
                cache_rows,
            )
        _mark_tasks_done(db, symbol=symbol, interval_minutes=interval, feature_version=int(settings.feature_version), up_to_open_time_ms=max_ot)

        metrics.precompute_tasks_processed_total.labels(SERVICE, symbol, str(interval)).inc(len(open_times))
        metrics.feature_compute_seconds.labels(SERVICE, symbol).observe(time.time() - t0)
        return len(open_times)
    except Exception as e:
        metrics.precompute_errors_total.labels(SERVICE, symbol, str(interval)).inc()
        _mark_tasks_error(db, symbol=symbol, interval_minutes=interval, feature_version=int(settings.feature_version), open_times=open_times, trace_id=trace_id, err=str(e))
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
        # NOTE: market_data is not versioned by feature_version.
        # Passing extra params will crash PyMySQL with:
        #   "not all arguments converted during string formatting"
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
                    enq = enqueue_precompute_tasks(db, symbol=symbol, interval_minutes=interval, open_times=open_times, trace_id=trace_id, feature_version=int(settings.feature_version))
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
            # NOTE: market_data is not versioned by feature_version.
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
            enq = enqueue_precompute_tasks(db, symbol=symbol, interval_minutes=interval, open_times=open_times, trace_id=trace_id, feature_version=int(settings.feature_version))
            metrics.precompute_tasks_enqueued_total.labels(SERVICE, symbol, str(interval)).inc(enq)

        # Compute lag based on cache
        last_cache = db.fetch_one(
            """
            SELECT open_time_ms FROM market_data_cache
            WHERE symbol=%s AND interval_minutes=%s AND feature_version=%s
            ORDER BY open_time_ms DESC LIMIT 1
            """,
            (symbol, interval, int(settings.feature_version)),
        )
        if last_cache:
            last_open_ms = int(last_cache["open_time_ms"])
            # lag measured as: now - bar_close_time
            close_ms = last_open_ms + interval_ms
            lag = int(max(0, now_ms() - close_ms))
            metrics.data_sync_lag_ms.labels(SERVICE, symbol, str(interval)).set(lag)

            # Telegram alert when lag exceeds threshold (throttled)
            try:
                threshold_ms = int(float(getattr(settings, 'market_data_lag_alert_seconds', 120)) * 1000.0)
                cooldown_s = float(getattr(settings, 'market_data_lag_alert_cooldown_seconds', 300))
                now_ts = time.time()
                last_ts = float(LAG_ALERT_LAST_AT.get(symbol, 0.0) or 0.0)
                if threshold_ms > 0 and lag > threshold_ms and settings.is_telegram_enabled() and (now_ts - last_ts) >= cooldown_s:
                    LAG_ALERT_LAST_AT[symbol] = now_ts
                    summary = build_system_summary(
                        event='DATA_LAG',
                        trace_id=trace_id,
                        level='WARN',
                        actor=SERVICE,
                        exchange=settings.exchange,
                        reason_code='DATA_LAG',
                        reason='market_data_cache lag exceeds threshold',
                        extra={
                            'symbol': symbol,
                            'interval_minutes': interval,
                            'feature_version': int(settings.feature_version),
                            'lag_ms': lag,
                            'threshold_ms': threshold_ms,
                            'last_open_time_ms': last_open_ms,
                            'last_close_time_ms': close_ms,
                        },
                    )
                    send_system_alert(telegram, title='DATA_LAG', summary_kv=summary, payload={})
            except Exception:
                pass

        # gap fill on recent history
        _fill_recent_gaps(db, ex, settings, metrics, symbol=symbol, trace_id=trace_id)

        upsert_heartbeat(
            db,
            instance_id,
            {"trace_id": trace_id, "status": "OK", "symbol": symbol, "inserted": inserted},
        )
    except Exception as e:
        metrics.data_sync_errors_total.labels(SERVICE).inc()
        try:
            append_error_event(
                db,
                trace_id=trace_id,
                service=SERVICE,
                exchange=settings.exchange,
                symbol=symbol,
                reason=f"sync_error: {str(e)[:200]}",
                payload={"error": str(e), "symbol": symbol, "interval": interval},
                reason_code="DATA_SYNC",
            )
        except Exception:
            pass
        upsert_heartbeat(db, instance_id, {"trace_id": trace_id, "status": "ERROR", "symbol": symbol, "error": str(e)})
        logger.exception(f"sync_error symbol={symbol} trace_id={trace_id} err={e}")
        telegram.send(f"[{SERVICE}] sync_error symbol={symbol} trace_id={trace_id} err={e}")


def main():
    settings = load_settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, Path(__file__).resolve().parents[2] / "migrations")

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

    runtime_cfg = RuntimeConfig.load(db, settings)
    # update runtime config metrics (best-effort)
    try:
        metrics.runtime_config_symbols_count.labels(SERVICE).set(len(runtime_cfg.symbols))
        metrics.runtime_config_last_refresh_ms.labels(SERVICE).set(runtime_cfg.last_refresh_ms)
    except Exception:
        pass

    symbols = list(runtime_cfg.symbols)
    logger.info(
        f"start service={SERVICE} exchange={settings.exchange} interval={settings.interval_minutes} "
        f"symbols={symbols} symbols_from_db={runtime_cfg.symbols_from_db} refresh_seconds={settings.runtime_config_refresh_seconds}"
    )

    next_cfg_refresh_ts = time.time() + float(settings.runtime_config_refresh_seconds)

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

        # Runtime config hot-reload (A2): refresh SYMBOLS/HALT/EMERGENCY flags
        if time.time() >= next_cfg_refresh_ts:
            try:
                changes = runtime_cfg.refresh(db, settings)
                metrics.runtime_config_refresh_total.labels(SERVICE).inc()
                metrics.runtime_config_symbols_count.labels(SERVICE).set(len(runtime_cfg.symbols))
                metrics.runtime_config_last_refresh_ms.labels(SERVICE).set(runtime_cfg.last_refresh_ms)
                if "symbols" in changes:
                    symbols = list(runtime_cfg.symbols)
                    logger.info(
                        f"runtime_config_symbols_updated symbols={symbols} symbols_from_db={runtime_cfg.symbols_from_db}"
                    )
            except Exception as e:
                logger.warning(f"runtime_config_refresh_failed err={e}")
            next_cfg_refresh_ts = time.time() + float(settings.runtime_config_refresh_seconds)

        metrics.data_sync_cycles_total.labels(SERVICE).inc()
        # daily archive
        run_daily_archive(db, settings, metrics, instance_id=instance_id)

        for sym in symbols:
            try:
                sync_symbol_once(db, ex, settings, metrics, telegram, symbol=sym, instance_id=instance_id)
            except RateLimitError as e:
                sleep_s = e.retry_after_seconds or 2.0
                try:
                    telegram.send(f"[RATE_LIMIT] group={e.group} sleep={sleep_s:.2f}s severe={e.severe} sym={sym}")
                except Exception:
                    pass
                time.sleep(max(0.5, float(sleep_s)))
                continue
            # process a slice of precompute tasks per symbol each loop
            processed = process_precompute_tasks(db, settings, metrics, symbol=sym, max_tasks=800)
            if processed:
                logger.info(f"precompute_done symbol={sym} processed={processed}")

        time.sleep(10)


if __name__ == "__main__":
    main()