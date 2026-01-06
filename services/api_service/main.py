
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.logging import get_logger, new_trace_id
from shared.telemetry import Telegram

SERVICE = "api-service"
VERSION = "v8.3-mvp"

app = FastAPI(title="Alpha-Sniper-V8 API", version=VERSION)
logger = get_logger(SERVICE, os.getenv("LOG_LEVEL", "INFO"))

def get_settings() -> Settings:
    return Settings()

def get_db(settings: Settings = Depends(get_settings)) -> MariaDB:
    return MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)

def auth_admin(authorization: Optional[str] = Header(default=None), settings: Settings = Depends(get_settings)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid token")

@app.on_event("startup")
def startup() -> None:
    settings = Settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    ran = migrate(db, Path("/app/migrations"))
    Telegram(settings.telegram_bot_token, settings.telegram_chat_id).send(f"[{SERVICE}] migrations: {', '.join(ran) if ran else 'none'}")

@app.get("/health")
def health(settings: Settings = Depends(get_settings), db: MariaDB = Depends(get_db)) -> Dict[str, Any]:
    return {"service": SERVICE, "version": VERSION, "env": settings.app_env, "exchange": settings.exchange, "symbol": settings.symbol, "db_ok": db.ping()}

@app.get("/metrics")
def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

def write_system_config(db: MariaDB, *, actor: str, key: str, value: str, trace_id: str, reason_code: str, reason: str) -> None:
    old = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
    old_val = old["value"] if old else None

    db.execute("INSERT INTO system_config(`key`,`value`) VALUES (%s,%s) ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)", (key, value))
    db.execute(
        """
        INSERT INTO config_audit(actor, action, cfg_key, old_value, new_value, trace_id, reason_code, reason)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (actor, "SET", key, old_val, value, trace_id, reason_code, reason),
    )

@app.post("/admin/halt", dependencies=[Depends(auth_admin)])
def admin_halt(payload: Dict[str, Any], settings: Settings = Depends(get_settings), db: MariaDB = Depends(get_db)):
    trace_id = new_trace_id("admin")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason is required")
    write_system_config(db, actor="api", key="HALT_TRADING", value="true", trace_id=trace_id, reason_code="ADMIN_HALT", reason=reason)
    Telegram(settings.telegram_bot_token, settings.telegram_chat_id).send(f"[HALT] trace_id={trace_id} reason={reason}")
    return {"ok": True, "trace_id": trace_id}

@app.post("/admin/resume", dependencies=[Depends(auth_admin)])
def admin_resume(payload: Dict[str, Any], settings: Settings = Depends(get_settings), db: MariaDB = Depends(get_db)):
    trace_id = new_trace_id("admin")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason is required")
    write_system_config(db, actor="api", key="HALT_TRADING", value="false", trace_id=trace_id, reason_code="ADMIN_HALT", reason=reason)
    Telegram(settings.telegram_bot_token, settings.telegram_chat_id).send(f"[RESUME] trace_id={trace_id} reason={reason}")
    return {"ok": True, "trace_id": trace_id}

@app.post("/admin/emergency_exit", dependencies=[Depends(auth_admin)])
def emergency_exit(payload: Dict[str, Any], settings: Settings = Depends(get_settings), db: MariaDB = Depends(get_db)):
    trace_id = new_trace_id("admin")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason is required")
    write_system_config(db, actor="api", key="EMERGENCY_EXIT", value="true", trace_id=trace_id, reason_code="EMERGENCY_EXIT", reason=reason)
    Telegram(settings.telegram_bot_token, settings.telegram_chat_id).send(f"[EMERGENCY_EXIT] trace_id={trace_id} reason={reason}")
    return {"ok": True, "trace_id": trace_id}

@app.post("/admin/update_config", dependencies=[Depends(auth_admin)])
def update_config(payload: Dict[str, Any], settings: Settings = Depends(get_settings), db: MariaDB = Depends(get_db)):
    trace_id = new_trace_id("admin")
    key = str(payload.get("key", "")).strip()
    value = str(payload.get("value", "")).strip()
    reason_code = str(payload.get("reason_code", "")).strip() or "SYSTEM"
    reason = str(payload.get("reason", "")).strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")
    if reason == "":
        raise HTTPException(status_code=400, detail="reason is required")
    write_system_config(db, actor="api", key=key, value=value, trace_id=trace_id, reason_code=reason_code, reason=reason)
    Telegram(settings.telegram_bot_token, settings.telegram_chat_id).send(f"[CONFIG] {key}={value} trace_id={trace_id} reason={reason}")
    return {"ok": True, "trace_id": trace_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.api_service.main:app", host="0.0.0.0", port=8080, reload=False)
