from __future__ import annotations

import os
from datetime import datetime, timezone
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.redis import redis_client
from shared.logging import get_logger, new_trace_id
from shared.domain.time import HK
from shared.telemetry import Telegram

SERVICE = "api-service"
VERSION = "0.1.0"

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


def get_system_config(db: MariaDB, key: str, default: Optional[str] = None) -> Optional[str]:
    row = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 推荐的生命周期事件（替代 on_event startup/shutdown）
    这里做：数据库迁移 + 启动告警（可选）
    """
    settings = Settings()
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    trace_id = new_trace_id("startup")

    try:
        db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
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
    yield

    # shutdown（可选）
    try:
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


def get_settings() -> Settings:
    return Settings()


def get_db(settings: Settings = Depends(get_settings)) -> MariaDB:
    return MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)


def require_admin(request: Request, authorization: str = Header(default=""), settings: Settings = Depends(get_settings)) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token not in set(settings.admin_tokens):
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
def health(settings: Settings = Depends(get_settings), db: MariaDB = Depends(get_db)) -> Dict[str, Any]:
    return {
        "service": SERVICE,
        "version": VERSION,
        "env": settings.env,
        "exchange": settings.exchange,
        "symbol": settings.symbol,
        "db_ok": db.ping(),
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


def write_system_config(
    db: MariaDB,
    *,
    actor: str,
    key: str,
    value: str,
    trace_id: str,
    reason_code: str,
    reason: str,
) -> None:
    old = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
    old_val = old["value"] if old else None

    db.execute(
        "INSERT INTO system_config(`key`,`value`) VALUES (%s,%s) ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)",
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
    db: MariaDB = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("status")

    halt_raw = get_system_config(db, "HALT_TRADING", "false")
    emergency_raw = get_system_config(db, "EMERGENCY_EXIT", "false")

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
        services[name] = {
            "instance_id": r["instance_id"],
            "last_heartbeat": str(r["last_heartbeat"]),
            "status": status_json,
        }

    # market data lag per symbol
    md_rows = db.fetch_all(
        """
        SELECT symbol, MAX(open_time_ms) AS last_open_time_ms
        FROM market_data_cache
        GROUP BY symbol
        """
    )
    now_ms = int(time.time() * 1000)
    data_lag: List[Dict[str, Any]] = []
    for r in md_rows or []:
        last_ot = int(r["last_open_time_ms"]) if r["last_open_time_ms"] is not None else None
        lag_ms = (now_ms - last_ot) if last_ot else None
        data_lag.append({"symbol": r["symbol"], "last_open_time_ms": last_ot, "lag_ms": lag_ms})

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

    return {
        "ok": True,
        "trace_id": trace_id,
        "config": {
            "HALT_TRADING": _parse_bool(halt_raw),
            "EMERGENCY_EXIT": _parse_bool(emergency_raw),
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
    db: MariaDB = Depends(get_db),
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
    db: MariaDB = Depends(get_db),
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
    db: MariaDB = Depends(get_db),
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
    db: MariaDB = Depends(get_db),
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
