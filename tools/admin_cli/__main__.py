
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.logging import new_trace_id
from shared.telemetry import Telegram

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

def main():
    settings = Settings()
    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    migrate(db, Path("migrations"))
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    p = argparse.ArgumentParser(prog="alpha-admin")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p_halt = sub.add_parser("halt")
    p_halt.add_argument("--reason", required=True)

    p_resume = sub.add_parser("resume")
    p_resume.add_argument("--reason", required=True)

    p_exit = sub.add_parser("emergency-exit")
    p_exit.add_argument("--reason", required=True)

    args = p.parse_args()
    if args.cmd == "status":
        rows = db.fetch_all("SELECT `key`,`value`,updated_at FROM system_config ORDER BY `key`")
        print(json.dumps(rows, ensure_ascii=False, default=str, indent=2))
        return

    trace_id = new_trace_id("cli")
    if args.cmd == "halt":
        write_system_config(db, actor="cli", key="HALT_TRADING", value="true", trace_id=trace_id, reason_code="ADMIN_HALT", reason=args.reason)
        telegram.send(f"[HALT] trace_id={trace_id} reason={args.reason}")
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "resume":
        write_system_config(db, actor="cli", key="HALT_TRADING", value="false", trace_id=trace_id, reason_code="ADMIN_HALT", reason=args.reason)
        telegram.send(f"[RESUME] trace_id={trace_id} reason={args.reason}")
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "emergency-exit":
        write_system_config(db, actor="cli", key="EMERGENCY_EXIT", value="true", trace_id=trace_id, reason_code="EMERGENCY_EXIT", reason=args.reason)
        telegram.send(f"[EMERGENCY_EXIT] trace_id={trace_id} reason={args.reason}")
        print(f"OK trace_id={trace_id}")
        return

if __name__ == "__main__":
    main()
