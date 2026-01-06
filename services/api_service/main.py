from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.logging import get_logger, new_trace_id
from shared.telemetry import Telegram

SERVICE = "api-service"
VERSION = "0.1.0"

app = FastAPI(title=SERVICE, version=VERSION)
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))


def tg_alert(telegram: Telegram, *, level: str, event: str, title: str, trace_id: str, summary_extra: dict, payload_extra: dict) -> None:
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
        "service": "api-service",
        "trace_id": trace_id,
        **(payload_extra or {}),
    }
    telegram.send_alert_zh(title=title, summary_kv=summary_kv, payload=payload)


def get_settings() -> Settings:
    return Settings()


def get_db(settings: Settings = Depends(get_settings)) -> MariaDB:
    return MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)


def require_admin(authorization: str = Header(default=""), settings: Settings = Depends(get_settings)) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid token")


@app.on_event("startup")
def startup() -> None:
    settings = Settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    ran = migrate(db, Path("/app/migrations"))
    trace_id = new_trace_id("startup")

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="INFO",
        event="MIGRATIONS",
        title="🧱 数据库迁移",
        trace_id=trace_id,
        summary_extra={"执行": (", ".join(ran) if ran else "无")},
        payload_extra={"ran": ran},
    )


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


@app.post("/admin/halt")
def admin_halt(
    payload: Dict[str, Any],
    settings: Settings = Depends(get_settings),
    db: MariaDB = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("halt")
    reason = str(payload.get("reason", "")).strip() or "admin_halt"
    write_system_config(db, actor="api", key="HALT_TRADING", value="true", trace_id=trace_id, reason_code="ADMIN_HALT", reason=reason)

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="WARN",
        event="ADMIN_HALT",
        title="⏸️ 管理操作：暂停交易",
        trace_id=trace_id,
        summary_extra={"原因": reason},
        payload_extra={"reason_code": "ADMIN_HALT", "key": "HALT_TRADING", "value": "true", "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}


@app.post("/admin/resume")
def admin_resume(
    payload: Dict[str, Any],
    settings: Settings = Depends(get_settings),
    db: MariaDB = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("resume")
    reason = str(payload.get("reason", "")).strip() or "admin_resume"
    write_system_config(db, actor="api", key="HALT_TRADING", value="false", trace_id=trace_id, reason_code="ADMIN_HALT", reason=reason)

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="INFO",
        event="ADMIN_RESUME",
        title="▶️ 管理操作：恢复交易",
        trace_id=trace_id,
        summary_extra={"原因": reason},
        payload_extra={"reason_code": "ADMIN_HALT", "key": "HALT_TRADING", "value": "false", "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}


@app.post("/admin/emergency_exit")
def admin_emergency_exit(
    payload: Dict[str, Any],
    settings: Settings = Depends(get_settings),
    db: MariaDB = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("exit")
    reason = str(payload.get("reason", "")).strip() or "admin_emergency_exit"
    write_system_config(db, actor="api", key="EMERGENCY_EXIT", value="true", trace_id=trace_id, reason_code="EMERGENCY_EXIT", reason=reason)

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="CRITICAL",
        event="ADMIN_EMERGENCY_EXIT",
        title="🆘 管理操作：紧急退出",
        trace_id=trace_id,
        summary_extra={"原因": reason},
        payload_extra={"reason_code": "EMERGENCY_EXIT", "key": "EMERGENCY_EXIT", "value": "true", "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}


@app.post("/admin/update_config")
def admin_update_config(
    payload: Dict[str, Any],
    settings: Settings = Depends(get_settings),
    db: MariaDB = Depends(get_db),
    _: None = Depends(require_admin),
) -> Dict[str, Any]:
    trace_id = new_trace_id("cfg")
    key = str(payload.get("key", "")).strip()
    value = str(payload.get("value", "")).strip()
    reason_code = str(payload.get("reason_code", "")).strip() or "ADMIN_CONFIG"
    reason = str(payload.get("reason", "")).strip() or "admin_update_config"

    if not key:
        raise HTTPException(status_code=400, detail="Missing key")

    write_system_config(db, actor="api", key=key, value=value, trace_id=trace_id, reason_code=reason_code, reason=reason)

    tg_alert(
        Telegram(settings.telegram_bot_token, settings.telegram_chat_id),
        level="INFO",
        event="ADMIN_UPDATE_CONFIG",
        title="⚙️ 管理操作：修改配置",
        trace_id=trace_id,
        summary_extra={"key": key, "value": value, "原因": reason},
        payload_extra={"reason_code": reason_code, "key": key, "value": value, "reason": reason},
    )
    return {"ok": True, "trace_id": trace_id}
