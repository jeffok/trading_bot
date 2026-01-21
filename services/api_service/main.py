from __future__ import annotations

import os
from datetime import datetime, timezone
import json
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse, HTMLResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from shared.config import Settings, load_settings, ALLOWED_EXCHANGES
from shared.db import PostgreSQL, migrate
from shared.redis import redis_client
from shared.logging import get_logger, new_trace_id
from shared.domain.time import HK
from shared.telemetry import Telegram, log_action
from shared.domain.control_commands import write_control_command
from shared.domain.heartbeat import upsert_service_status
from shared.domain.instance import get_instance_id
from shared.domain.events import append_error_event

SERVICE = "api-service"
VERSION = "0.1.1"

logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))
# ===== Admin models (V8.3 hard requirement: actor + reason_code + reason) =====
class AdminMeta(BaseModel):
    actor: str = Field(..., min_length=1, max_length=64, description="操作人/来源（必须）")
    reason_code: str = Field(..., min_length=1, max_length=64, description="原因代码（必须）")
    reason: str = Field(..., min_length=1, max_length=4096, description="原因说明（必须）")
    confirm_code: Optional[str] = Field(default=None, max_length=128, description="二次确认码（可选，开启时必填）")


class AdminUpdateConfig(AdminMeta):
    key: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., min_length=0, max_length=4096)


def _parse_bool(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _parse_symbols_list(raw: str) -> list[str]:
    import re as _re
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = []
    for token in _re.split(r"[\s,]+", raw):
        t = token.strip().upper()
        if t:
            parts.append(t)
    seen = set()
    out = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out



def get_system_config(db: PostgreSQL, key: str, default: Optional[str] = None) -> Optional[str]:
    row = db.fetch_one('SELECT "value" FROM system_config WHERE "key"=%s', (key,))
    return row["value"] if row else default


def expected_reason_code(cmd_reason_code: str, expected: str) -> None:
    # 强制 reason_code 标准化，避免审计数据碎片化
    if cmd_reason_code != expected:
        raise HTTPException(status_code=400, detail=f"reason_code must be '{expected}'")


def require_confirm(cmd: AdminMeta, settings: Settings) -> None:
    if not settings.admin_confirm_required:
        return
    if not settings.admin_confirm_code:
        raise HTTPException(status_code=500, detail="ADMIN_CONFIRM_REQUIRED is enabled but ADMIN_CONFIRM_CODE is empty")
    if not cmd.confirm_code or cmd.confirm_code != settings.admin_confirm_code:
        raise HTTPException(status_code=400, detail="confirm_code required")




def tg_alert(
    telegram: Telegram,
    *,
    level: str,
    event: str,
    title: str,
    trace_id: str,
    summary_extra: dict,
    payload_extra: dict,
) -> None:
    """
    统一告警封装：只负责发 Telegram（展示中文化由 Telegram.send_alert_zh 处理）
    """
    summary_kv = {
        "level": level,
        "event": event,
        "service": "管理接口",
        "trace_id": trace_id,
        **(summary_extra or {}),
    }
    payload = {
        "level": level,
        "event": event,
        "service": SERVICE,
        "trace_id": trace_id,
        **(payload_extra or {}),
    }

    # 兼容：如果你项目里 Telegram 还没 send_alert_zh，就退回 send_alert
    if hasattr(telegram, "send_alert_zh"):
        telegram.send_alert_zh(title=title, summary_kv=summary_kv, payload=payload)
    else:
        telegram.send_alert(title=title, summary_lines=[f"{k}={v}" for k, v in summary_kv.items()], payload=payload)



    try:
        log_action(logger, event, trace_id=trace_id, level=level, title=title, **(summary_extra or {}))
    except Exception:
        pass
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 推荐的生命周期事件（替代 on_event startup/shutdown）
    这里做：数据库迁移 + 启动告警（可选）
    """
    settings = load_settings()
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    trace_id = new_trace_id("startup")

    try:
        db = PostgreSQL(settings.postgres_url)
        ran = migrate(db, Path("/app/migrations"))

        tg_alert(
            telegram,
            level="INFO",
            event="MIGRATIONS",
            title="🧱 数据库迁移完成",
            trace_id=trace_id,
            summary_extra={"执行": (", ".join(ran) if ran else "无")},
            payload_extra={"ran": ran},
        )

        logger.info("startup ok: migrations=%s", ran)

    except Exception as e:
        # 迁移失败就直接抛出，让服务启动失败（这是正确行为）
        logger.exception("startup failed (migrations)")
        tg_alert(
            telegram,
            level="ERROR",
            event="MIGRATIONS_FAILED",
            title="❌ 数据库迁移失败，服务启动终止",
            trace_id=trace_id,
            summary_extra={"错误": str(e)[:200]},
            payload_extra={"error": str(e)},
        )
        raise

    # 进入运行期

    # API 服务心跳（写入 service_status）
    stop_evt = threading.Event()
    instance_id = get_instance_id(SERVICE, settings.instance_id)

    def _hb_loop() -> None:
        db = PostgreSQL(settings.postgres_url)
        started = time.time()
        while not stop_evt.is_set():
            try:
                status = {
                    "service": SERVICE,
                    "version": VERSION,
                    "env": settings.env,
                    "exchange": settings.exchange,
                    "symbol": settings.symbol,
                    "uptime_sec": int(time.time() - started),
                    "pid": os.getpid(),
                }
                upsert_service_status(db, service_name=SERVICE, instance_id=instance_id, status=status)
            except Exception:
                # 心跳失败不应导致服务退出
                logger.exception("heartbeat failed")
            stop_evt.wait(max(5, int(getattr(settings, "heartbeat_interval_seconds", 30))))
    api_heartbeat_thread = threading.Thread(target=_hb_loop, name="api-hb", daemon=True)
    api_heartbeat_thread.start()

    yield

    # shutdown（可选）
    try:
        stop_evt.set()
        try:
            api_heartbeat_thread.join(timeout=2)
        except Exception:
            pass
        trace_id2 = new_trace_id("shutdown")
        tg_alert(
            telegram,
            level="INFO",
            event="SHUTDOWN",
            title="🛑 服务停止",
            trace_id=trace_id2,
            summary_extra={},
            payload_extra={},
        )
    except Exception:
        pass


app = FastAPI(title=SERVICE, version=VERSION, lifespan=lifespan)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    settings = load_settings()
    trace_id = new_trace_id("api_exc")
    try:
        db = PostgreSQL(settings.postgres_url)
        # use first effective symbol if available
        sym = (list(getattr(settings, "symbols", ()) or []) + [getattr(settings, "symbol", "")])[0] or "UNKNOWN"
        append_error_event(
            db,
            trace_id=trace_id,
            service=SERVICE,
            exchange=settings.exchange,
            symbol=str(sym),
            reason=f"api_exception: {str(exc)[:200]}",
            payload={
                "path": str(request.url.path),
                "method": str(request.method),
                "error": str(exc),
            },
            reason_code="SYSTEM",
        )
    except Exception:
        pass
    logger.exception(f"unhandled_exception trace_id={trace_id} err={exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "trace_id": trace_id})


def get_settings() -> Settings:
    return load_settings()


def get_db(settings: Settings = Depends(get_settings)) -> PostgreSQL:
    return PostgreSQL(settings.postgres_url)


def require_admin(request: Request, authorization: str = Header(default=""), settings: Settings = Depends(get_settings)) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid token")

    # IP allowlist (optional)
    if settings.admin_ip_allowlist:
        xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
        client_ip = ""
        if xff:
            client_ip = xff.split(",", 1)[0].strip()
        elif request.client:
            client_ip = request.client.host or ""
        if not is_ip_allowed(client_ip, settings.admin_ip_allowlist):
            raise HTTPException(status_code=403, detail="IP not allowed")


@app.get("/health")
def health(settings: Settings = Depends(get_settings), db: PostgreSQL = Depends(get_db)) -> Dict[str, Any]:
    """Lightweight health endpoint (no admin auth).
    Includes: db ping, halt/emergency flags, last heartbeats, and market data lag for effective symbols.
    """
    now_utc = datetime.now(timezone.utc)
    now_hk = now_utc.astimezone(HK)

    # runtime flags
    halt_raw = get_system_config(db, "HALT_TRADING", "false")
    emergency_raw = get_system_config(db, "EMERGENCY_EXIT", "false")

    # services latest heartbeat snapshot
    rows = db.fetch_all(
        """
        SELECT service_name, instance_id, last_heartbeat, status_json
        FROM service_status
        ORDER BY last_heartbeat DESC
        LIMIT 50
        """
    )
    services: Dict[str, Any] = {}
    for r in rows or []:
        name = r["service_name"]
        if name in services:
            continue
        try:
            status_json = json.loads(r["status_json"]) if isinstance(r["status_json"], str) else r["status_json"]
        except Exception:
            status_json = {"raw": r["status_json"]}
        
        # 处理时间戳：转换为 ISO 格式（带时区），便于前端解析
        # 数据库存储的是 UTC 时间，需要明确标注时区
        last_heartbeat = r["last_heartbeat"]
        if isinstance(last_heartbeat, datetime):
            # 如果是 datetime 对象，转换为 ISO 格式字符串（UTC）
            if last_heartbeat.tzinfo is None:
                # 如果没有时区信息，假设是 UTC
                last_heartbeat_utc = last_heartbeat.replace(tzinfo=timezone.utc)
            else:
                last_heartbeat_utc = last_heartbeat
            last_heartbeat_str = last_heartbeat_utc.isoformat()
        else:
            # 如果是字符串，尝试解析并转换为 ISO 格式
            try:
                # PostgreSQL 返回的格式可能是 '2026-01-18 11:43:12'
                # 假设是 UTC 时间
                if isinstance(last_heartbeat, str) and ' ' in last_heartbeat:
                    dt_str = last_heartbeat.replace(' ', 'T')
                    if '+' not in dt_str and 'Z' not in dt_str:
                        dt_str += '+00:00'  # 添加 UTC 时区
                    last_heartbeat_str = dt_str
                else:
                    last_heartbeat_str = str(last_heartbeat)
            except Exception:
                last_heartbeat_str = str(last_heartbeat)
        
        services[name] = {
            "instance_id": r["instance_id"],
            "last_heartbeat": last_heartbeat_str,
            "status": status_json,
        }

    base_syms = list(getattr(settings, "symbols", ()) or [])
    if not base_syms:
        base_syms = [settings.symbol]
    effective_symbols = _normalize_symbols(base_syms)

    # market data lag (only effective symbols)
    now_ms = int(time.time() * 1000)
    data_lag: List[Dict[str, Any]] = []
    for sym in effective_symbols:
        r = db.fetch_one(
            "SELECT MAX(open_time_ms) AS last_open_time_ms FROM market_data_cache WHERE symbol=%s AND interval_minutes=%s AND feature_version=%s",
            (sym, int(settings.interval_minutes), int(settings.feature_version)),
        )
        last_ot = int(r["last_open_time_ms"]) if r and r.get("last_open_time_ms") is not None else None
        lag_ms = (now_ms - last_ot) if last_ot else None
        data_lag.append({"symbol": sym, "last_open_time_ms": last_ot, "lag_ms": lag_ms})


    # engine last tick (best-effort)
    engine_last_tick: Dict[str, Any] = {}
    try:
        se = services.get("strategy-engine") or services.get("strategy_engine")
        if se and isinstance(se.get("status"), dict):
            st = se.get("status")
            engine_last_tick = {
                "last_tick_id": st.get("last_tick_id"),
                "last_tick_ts_utc": st.get("last_tick_ts_utc"),
                "last_tick_ts_hk": st.get("last_tick_ts_hk"),
                "trace_id": st.get("trace_id"),
            }
    except Exception:
        engine_last_tick = {}

    # recent errors summary (best-effort)
    recent_errors: List[Dict[str, Any]] = []
    try:
        err_rows = db.fetch_all(
            """
            SELECT id, created_at, trace_id, service, exchange, symbol, client_order_id, reason_code, reason
            FROM order_events
            WHERE event_type='ERROR'
            ORDER BY id DESC
            LIMIT 10
            """
        )
        for r in err_rows or []:
            recent_errors.append(
                {
                    "id": int(r.get("id") or 0),
                    "created_at": str(r.get("created_at")),
                    "trace_id": r.get("trace_id"),
                    "service": r.get("service"),
                    "exchange": r.get("exchange"),
                    "symbol": r.get("symbol"),
                    "client_order_id": r.get("client_order_id"),
                    "reason_code": r.get("reason_code"),
                    "reason": (str(r.get("reason") or "")[:200]),
                }
            )
    except Exception:
        recent_errors = []
    return {
        "service": SERVICE,
        "version": VERSION,
        "env": settings.env,
        "exchange": settings.exchange,
        "symbols": effective_symbols,
        "now_utc": now_utc.isoformat(),
        "now_hk": now_hk.isoformat(),
        "db_ok": db.ping(),
        "halt_trading": _parse_bool(halt_raw),
        "emergency_exit": _parse_bool(emergency_raw),
        "services": services,
        "market_data_lag": data_lag,
        "engine_last_tick": engine_last_tick,
        "recent_errors": recent_errors,
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/ui", response_class=HTMLResponse)
def admin_ui() -> HTMLResponse:
    """Web管理界面"""
    ui_file = Path(__file__).parent / "admin_ui.html"
    if not ui_file.exists():
        return HTMLResponse(content="<h1>管理界面文件未找到</h1>", status_code=404)
    return FileResponse(ui_file)


def write_system_config(
    db: PostgreSQL,
    *,
    actor: str,
    key: str,
    value: str,
    trace_id: str,
    reason_code: str,
    reason: str,
) -> None:
    old = db.fetch_one('SELECT "value" FROM system_config WHERE "key"=%s', (key,))
    old_val = old["value"] if old else None

    db.execute(
        'INSERT INTO system_config("key","value") VALUES (%s,%s) ON CONFLICT ("key") DO UPDATE SET "value"=EXCLUDED."value"',
        (key, value),
    )
    db.execute(
        """
        INSERT INTO config_audit(actor, action, cfg_key, old_value, new_value, trace_id, reason_code, reason)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (actor, "SET", key, old_val, value, trace_id, reason_code, reason),
    )


@app.get("/admin/status")
def admin_status(
    settings: Settings = Depends(get_settings),
    db: PostgreSQL = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("status")

    halt_raw = get_system_config(db, "HALT_TRADING", "false")
    emergency_raw = get_system_config(db, "EMERGENCY_EXIT", "false")

    use_stop_raw = get_system_config(db, "USE_PROTECTIVE_STOP_ORDER", "true" if getattr(settings, "use_protective_stop_order", True) else "false")
    stop_poll_raw = get_system_config(db, "STOP_ORDER_POLL_SECONDS", str(getattr(settings, "stop_order_poll_seconds", 10)))
    stop_arm_retries_raw = get_system_config(db, "STOP_ARM_MAX_RETRIES", str(getattr(settings, "stop_arm_max_retries", 3)))
    stop_arm_backoff_raw = get_system_config(db, "STOP_ARM_BACKOFF_BASE_SECONDS", str(getattr(settings, "stop_arm_backoff_base_seconds", 0.5)))
    stop_rearm_max_raw = get_system_config(db, "STOP_REARM_MAX_ATTEMPTS", str(getattr(settings, "stop_rearm_max_attempts", 2)))
    stop_rearm_cd_raw = get_system_config(db, "STOP_REARM_COOLDOWN_SECONDS", str(getattr(settings, "stop_rearm_cooldown_seconds", 60)))
    use_protective_stop_order = _parse_bool(use_stop_raw)
    try:
        stop_order_poll_seconds = int(float(stop_poll_raw))
    except Exception:
        stop_order_poll_seconds = int(getattr(settings, "stop_order_poll_seconds", 10))
    try:
        stop_arm_max_retries = int(float(stop_arm_retries_raw))
    except Exception:
        stop_arm_max_retries = int(getattr(settings, "stop_arm_max_retries", 3))

    try:
        stop_arm_backoff_base_seconds = float(stop_arm_backoff_raw)
    except Exception:
        stop_arm_backoff_base_seconds = float(getattr(settings, "stop_arm_backoff_base_seconds", 0.5))

    try:
        stop_rearm_max_attempts = int(float(stop_rearm_max_raw))
    except Exception:
        stop_rearm_max_attempts = int(getattr(settings, "stop_rearm_max_attempts", 2))

    try:
        stop_rearm_cooldown_seconds = int(float(stop_rearm_cd_raw))
    except Exception:
        stop_rearm_cooldown_seconds = int(getattr(settings, "stop_rearm_cooldown_seconds", 60))

    symbols_db_raw = get_system_config(db, "SYMBOLS", "")
    symbols_db = _parse_symbols_list(symbols_db_raw)
    env_symbols = list(settings.symbols) if getattr(settings, "symbols", None) else [settings.symbol]
    effective_symbols = symbols_db if symbols_db else env_symbols
    symbols_from_db = bool(symbols_db)

    # latest heartbeat per service (if any)
    rows = db.fetch_all(
        """
        SELECT service_name, instance_id, last_heartbeat, status_json
        FROM service_status
        ORDER BY last_heartbeat DESC
        """
    )
    services: Dict[str, Any] = {}
    for r in rows or []:
        name = r["service_name"]
        if name in services:
            continue
        try:
            status_json = json.loads(r["status_json"]) if isinstance(r["status_json"], str) else r["status_json"]
        except Exception:
            status_json = {"raw": r["status_json"]}
        
        # 处理时间戳：转换为 ISO 格式（带时区），便于前端解析
        # 数据库存储的是 UTC 时间，需要明确标注时区
        last_heartbeat = r["last_heartbeat"]
        if isinstance(last_heartbeat, datetime):
            # 如果是 datetime 对象，转换为 ISO 格式字符串（UTC）
            if last_heartbeat.tzinfo is None:
                # 如果没有时区信息，假设是 UTC
                last_heartbeat_utc = last_heartbeat.replace(tzinfo=timezone.utc)
            else:
                last_heartbeat_utc = last_heartbeat
            last_heartbeat_str = last_heartbeat_utc.isoformat()
        else:
            # 如果是字符串，尝试解析并转换为 ISO 格式
            try:
                # PostgreSQL 返回的格式可能是 '2026-01-18 11:43:12'
                # 假设是 UTC 时间
                if isinstance(last_heartbeat, str) and ' ' in last_heartbeat:
                    dt_str = last_heartbeat.replace(' ', 'T')
                    if '+' not in dt_str and 'Z' not in dt_str:
                        dt_str += '+00:00'  # 添加 UTC 时区
                    last_heartbeat_str = dt_str
                else:
                    last_heartbeat_str = str(last_heartbeat)
            except Exception:
                last_heartbeat_str = str(last_heartbeat)
        
        services[name] = {
            "instance_id": r["instance_id"],
            "last_heartbeat": last_heartbeat_str,
            "status": status_json,
        }

    # market data lag per symbol and latest price
    # 使用 market_data 表的最新数据（WebSocket 实时数据）
    # 延迟计算：now_ms - (last_open_time_ms + interval_ms) = now_ms - close_time_ms
    interval_minutes = int(settings.interval_minutes)
    interval_ms = interval_minutes * 60_000
    feature_version = int(settings.feature_version)
    
    # 优先使用 market_data 表（WebSocket 实时数据），获取最新 K线
    # 如果没有 market_data，则使用 market_data_cache
    effective_symbols_list = list(effective_symbols) if effective_symbols else []
    if not effective_symbols_list:
        effective_symbols_list = ["BTCUSDT", "ETHUSDT"]  # 默认交易对
    
    md_rows = db.fetch_all(
        """
        SELECT DISTINCT
            COALESCE(m.symbol, s.symbol) AS symbol,
            COALESCE(m.open_time_ms, c.last_open_time_ms) AS last_open_time_ms,
            m.close_price AS latest_price
        FROM (
            SELECT unnest(%s::text[]) AS symbol
        ) s
        LEFT JOIN (
            SELECT symbol, open_time_ms, close_price
            FROM market_data
            WHERE interval_minutes = %s
            AND (symbol, open_time_ms) IN (
                SELECT symbol, MAX(open_time_ms)
                FROM market_data
                WHERE interval_minutes = %s
                GROUP BY symbol
            )
        ) m ON m.symbol = s.symbol
        LEFT JOIN (
            SELECT symbol, MAX(open_time_ms) AS last_open_time_ms
            FROM market_data_cache
            WHERE interval_minutes=%s AND feature_version=%s
            GROUP BY symbol
        ) c ON c.symbol = s.symbol
        """,
        (effective_symbols_list, interval_minutes, interval_minutes, interval_minutes, feature_version),
    )
    now_ms = int(time.time() * 1000)
    data_lag: List[Dict[str, Any]] = []
    
    # 尝试从 Redis 获取最新价格（价格服务的实时缓存）
    price_cache: Dict[str, float] = {}
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r_client = redis_client(redis_url)
        for symbol in effective_symbols_list:
            key = f"market:price:{symbol}"
            cached = r_client.get(key)
            if cached:
                try:
                    price_data = json.loads(cached)
                    price_cache[symbol] = float(price_data.get("price", 0))
                except Exception:
                    pass
    except Exception:
        pass  # Redis 不可用时忽略，使用数据库价格
    
    for r in md_rows or []:
        symbol = r["symbol"]
        last_ot = int(r["last_open_time_ms"]) if r["last_open_time_ms"] is not None else None
        # 正确的延迟计算：now_ms - (open_time_ms + interval_ms) = now_ms - close_time_ms
        # 或者简化为：now_ms - last_ot - interval_ms
        if last_ot:
            # 计算 K线收盘时间：open_time_ms + interval_ms
            close_time_ms = last_ot + interval_ms
            lag_ms = max(0, now_ms - close_time_ms)
        else:
            lag_ms = None
        
        # 优先使用 Redis 缓存的价格（实时），如果没有则使用数据库中的 K线收盘价
        latest_price = price_cache.get(symbol)
        if latest_price is None:
            latest_price = float(r["latest_price"]) if r["latest_price"] is not None else None
        
        data_lag.append({
            "symbol": symbol, 
            "last_open_time_ms": last_ot, 
            "lag_ms": lag_ms,
            "latest_price": latest_price
        })

    # open positions: latest snapshot per symbol base_qty>0
    pos_rows = db.fetch_all(
        """
        SELECT ps.symbol, ps.base_qty
        FROM position_snapshots ps
        JOIN (
            SELECT symbol, MAX(id) AS mid
            FROM position_snapshots
            GROUP BY symbol
        ) t ON ps.symbol=t.symbol AND ps.id=t.mid
        """
    )
    open_positions = 0
    positions: List[Dict[str, Any]] = []
    for r in pos_rows or []:
        qty = float(r["base_qty"] or 0)
        positions.append({"symbol": r["symbol"], "base_qty": qty})
        if qty > 0:
            open_positions += 1

    # 加载策略、风控、AI参数
    setup_b_adx_min_raw = get_system_config(db, "SETUP_B_ADX_MIN", "20.0")
    setup_b_vol_ratio_min_raw = get_system_config(db, "SETUP_B_VOL_RATIO_MIN", "1.5")
    setup_b_ai_score_min_raw = get_system_config(db, "SETUP_B_AI_SCORE_MIN", "55.0")
    hard_stop_loss_pct_raw = get_system_config(db, "HARD_STOP_LOSS_PCT", "0.03")
    account_equity_usdt_raw = get_system_config(db, "ACCOUNT_EQUITY_USDT", "500.0")
    risk_budget_pct_raw = get_system_config(db, "RISK_BUDGET_PCT", "0.03")
    max_drawdown_pct_raw = get_system_config(db, "MAX_DRAWDOWN_PCT", "0.15")
    max_concurrent_positions_raw = get_system_config(db, "MAX_CONCURRENT_POSITIONS", "3")
    min_order_usdt_raw = get_system_config(db, "MIN_ORDER_USDT", "50.0")
    ai_enabled_raw = get_system_config(db, "AI_ENABLED", "true")
    ai_weight_raw = get_system_config(db, "AI_WEIGHT", "0.35")
    ai_lr_raw = get_system_config(db, "AI_LR", "0.05")
    ai_min_samples_raw = get_system_config(db, "AI_MIN_SAMPLES", "50")

    return {
        "ok": True,
        "trace_id": trace_id,
        "config": {
            "EXCHANGE": settings.exchange,
            "SUPPORTED_EXCHANGES": sorted(list(ALLOWED_EXCHANGES)),
            "HALT_TRADING": _parse_bool(halt_raw),
            "EMERGENCY_EXIT": _parse_bool(emergency_raw),
            "EFFECTIVE_SYMBOLS": effective_symbols,
            "SYMBOLS_FROM_DB": symbols_from_db,
            "USE_PROTECTIVE_STOP_ORDER": bool(use_protective_stop_order),
            "STOP_ORDER_POLL_SECONDS": int(stop_order_poll_seconds),
            "STOP_ARM_MAX_RETRIES": int(stop_arm_max_retries),
            "STOP_ARM_BACKOFF_BASE_SECONDS": float(stop_arm_backoff_base_seconds),
            "STOP_REARM_MAX_ATTEMPTS": int(stop_rearm_max_attempts),
            "STOP_REARM_COOLDOWN_SECONDS": int(stop_rearm_cooldown_seconds),
            # 策略参数
            "SETUP_B_ADX_MIN": float(setup_b_adx_min_raw),
            "SETUP_B_VOL_RATIO_MIN": float(setup_b_vol_ratio_min_raw),
            "SETUP_B_AI_SCORE_MIN": float(setup_b_ai_score_min_raw),
            "HARD_STOP_LOSS_PCT": float(hard_stop_loss_pct_raw),
            # 风控参数
            "ACCOUNT_EQUITY_USDT": float(account_equity_usdt_raw),
            "RISK_BUDGET_PCT": float(risk_budget_pct_raw),
            "MAX_DRAWDOWN_PCT": float(max_drawdown_pct_raw),
            "MAX_CONCURRENT_POSITIONS": int(max_concurrent_positions_raw),
            "MIN_ORDER_USDT": float(min_order_usdt_raw),
            # AI参数
            "AI_ENABLED": _parse_bool(ai_enabled_raw),
            "AI_WEIGHT": float(ai_weight_raw),
            "AI_LR": float(ai_lr_raw),
            "AI_MIN_SAMPLES": int(ai_min_samples_raw),
            # SYMBOLS（用于显示当前值）
            "SYMBOLS": symbols_db_raw if symbols_db_raw else "",
        },
        "open_positions": open_positions,
        "positions": positions,
        "data_lag": data_lag,
        "services": services,
        "security": {
            "admin_ip_allowlist_enabled": bool(settings.admin_ip_allowlist),
            "admin_confirm_required": bool(settings.admin_confirm_required),
            "leader_election_enabled": bool(settings.leader_election_enabled),
        },
    }

@app.post("/admin/halt")
def admin_halt(
    cmd: AdminMeta,
    settings: Settings = Depends(get_settings),
    db: PostgreSQL = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("halt")
    expected_reason_code(cmd.reason_code, "ADMIN_HALT")
    reason = cmd.reason

    write_system_config(
        db,
        actor=cmd.actor,
        key="HALT_TRADING",
        value="true",
        trace_id=trace_id,
        reason_code=cmd.reason_code,
        reason=reason,
    )
    # audit queue (control_commands)
    write_control_command(
        db,
        command="HALT",
        payload={"actor": cmd.actor, "reason_code": cmd.reason_code, "reason": cmd.reason, "trace_id": trace_id},
        trace_id=trace_id,
        actor=cmd.actor,
        reason_code=cmd.reason_code,
        reason=cmd.reason,
    )

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="WARN",
        event="ADMIN_HALT",
        title="⏸️ 管理操作：暂停交易",
        trace_id=trace_id,
        summary_extra={"原因": reason},
        payload_extra={"reason_code": cmd.reason_code, "key": "HALT_TRADING", "value": "true", "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}


@app.post("/admin/resume")
def admin_resume(
    cmd: AdminMeta,
    settings: Settings = Depends(get_settings),
    db: PostgreSQL = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("resume")
    expected_reason_code(cmd.reason_code, "ADMIN_RESUME")
    reason = cmd.reason

    write_system_config(
        db,
        actor=cmd.actor,
        key="HALT_TRADING",
        value="false",
        trace_id=trace_id,
        reason_code=cmd.reason_code,
        reason=reason,
    )
    write_control_command(
        db,
        command="RESUME",
        payload={"actor": cmd.actor, "reason_code": cmd.reason_code, "reason": cmd.reason, "trace_id": trace_id},
        trace_id=trace_id,
        actor=cmd.actor,
        reason_code=cmd.reason_code,
        reason=cmd.reason,
    )

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="INFO",
        event="ADMIN_RESUME",
        title="▶️ 管理操作：恢复交易",
        trace_id=trace_id,
        summary_extra={"原因": reason},
        payload_extra={"reason_code": cmd.reason_code, "key": "HALT_TRADING", "value": "false", "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}


@app.post("/admin/emergency_exit")
def admin_emergency_exit(
    cmd: AdminMeta,
    settings: Settings = Depends(get_settings),
    db: PostgreSQL = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("exit")
    expected_reason_code(cmd.reason_code, "EMERGENCY_EXIT")
    require_confirm(cmd, settings)
    reason = cmd.reason

    write_system_config(
        db,
        actor=cmd.actor,
        key="EMERGENCY_EXIT",
        value="true",
        trace_id=trace_id,
        reason_code=cmd.reason_code,
        reason=reason,
    )
    write_control_command(
        db,
        command="EMERGENCY_EXIT",
        payload={"actor": cmd.actor, "reason_code": cmd.reason_code, "reason": cmd.reason, "trace_id": trace_id},
        trace_id=trace_id,
        actor=cmd.actor,
        reason_code=cmd.reason_code,
        reason=cmd.reason,
    )

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="CRITICAL",
        event="ADMIN_EMERGENCY_EXIT",
        title="🆘 管理操作：紧急退出",
        trace_id=trace_id,
        summary_extra={"原因": reason},
        payload_extra={"reason_code": cmd.reason_code, "key": "EMERGENCY_EXIT", "value": "true", "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}


@app.post("/admin/update_config")
def admin_update_config(
    cmd: AdminUpdateConfig,
    settings: Settings = Depends(get_settings),
    db: PostgreSQL = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("cfg")
    expected_reason_code(cmd.reason_code, "ADMIN_UPDATE_CONFIG")
    require_confirm(cmd, settings)
    key = cmd.key.strip()
    value = cmd.value
    reason = cmd.reason

    if not key:
        raise HTTPException(status_code=400, detail="Missing key")

    write_system_config(
        db,
        actor=cmd.actor,
        key=key,
        value=value,
        trace_id=trace_id,
        reason_code=cmd.reason_code,
        reason=reason,
    )
    write_control_command(
        db,
        command="UPDATE_CONFIG",
        payload={"actor": cmd.actor, "key": cmd.key, "value": cmd.value, "reason_code": cmd.reason_code, "reason": cmd.reason, "trace_id": trace_id},
        trace_id=trace_id,
        actor=cmd.actor,
        reason_code=cmd.reason_code,
        reason=cmd.reason,
    )

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="INFO",
        event="ADMIN_UPDATE_CONFIG",
        title="⚙️ 管理操作：修改配置",
        trace_id=trace_id,
        summary_extra={"key": key, "value": value, "原因": reason},
        payload_extra={"reason_code": cmd.reason_code, "key": key, "value": value, "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}


# ===== 回测相关 API =====

class BacktestRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="交易对")
    months: int = Field(default=6, description="回测月数")
    interval_minutes: Optional[int] = Field(default=None, description="K线周期（分钟）")
    feature_version: Optional[int] = Field(default=None, description="特征版本")
    initial_equity_usdt: float = Field(default=1000.0, description="初始资金USDT")
    fee_rate: float = Field(default=0.0004, description="手续费率")
    slippage_rate: float = Field(default=0.001, description="滑点率")


class BacktestIndividualRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="交易对")
    months: int = Field(default=6, description="回测月数")
    interval_minutes: Optional[int] = Field(default=None, description="K线周期（分钟）")
    conditions: Optional[List[str]] = Field(default=None, description="要测试的条件列表")
    test_all: bool = Field(default=False, description="测试所有5个条件")


@app.post("/admin/backtest-with-pnl")
def admin_backtest_with_pnl(
    req: BacktestRequest,
    settings: Settings = Depends(get_settings),
    db: PostgreSQL = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    """回测工具（带盈利计算）"""
    trace_id = new_trace_id("backtest_pnl")
    
    try:
        from scripts.trading_test_tool.backtest_with_pnl import run_backtest_with_pnl
        
        # 直接执行（同步），日志会输出到 stderr，可以通过 docker logs 查看
        result = run_backtest_with_pnl(
            symbol=req.symbol,
            months=req.months,
            interval_minutes=req.interval_minutes,
            initial_equity_usdt=req.initial_equity_usdt,
            fee_rate=req.fee_rate,
            slippage_rate=req.slippage_rate,
        )
        
        return {"ok": True, "exit_code": result, "trace_id": trace_id}
        
    except Exception as e:
        logger.exception(f"backtest_with_pnl failed trace_id={trace_id}")
        return {"ok": False, "error": str(e), "trace_id": trace_id}


@app.post("/admin/backtest-individual")
def admin_backtest_individual(
    req: BacktestIndividualRequest,
    settings: Settings = Depends(get_settings),
    db: PostgreSQL = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    """单独测试Setup B各条件"""
    trace_id = new_trace_id("backtest_individual")
    
    try:
        from scripts.trading_test_tool.backtest_individual_signals import run_individual_signals_test
        
        # 直接执行（同步），日志会输出到 stderr，可以通过 docker logs 查看
        result = run_individual_signals_test(
            symbol=req.symbol,
            months=req.months,
            interval_minutes=req.interval_minutes,
            condition_names=req.conditions,
            test_all=req.test_all,
        )
        
        return {"ok": True, "exit_code": result, "trace_id": trace_id}
        
    except Exception as e:
        logger.exception(f"backtest_individual failed trace_id={trace_id}")
        return {"ok": False, "error": str(e), "trace_id": trace_id}
