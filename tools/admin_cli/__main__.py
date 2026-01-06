from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.config import Settings
from shared.db import MariaDB, migrate
from shared.logging import new_trace_id
from shared.telemetry import Telegram

from .smoke import run_smoke_test


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

    p_smoke = sub.add_parser("smoke-test")
    p_smoke.add_argument("--wait-data-seconds", type=int, default=60)
    p_smoke.add_argument("--wait-engine-seconds", type=int, default=30)

    args = p.parse_args()

    if args.cmd == "status":
        rows = db.fetch_all("SELECT `key`,`value`,updated_at FROM system_config ORDER BY `key`")
        print(json.dumps(rows, ensure_ascii=False, default=str, indent=2))
        return

    if args.cmd == "smoke-test":
        report = run_smoke_test(wait_data_seconds=int(args.wait_data_seconds), wait_engine_seconds=int(args.wait_engine_seconds))
        print(json.dumps(report, ensure_ascii=False, default=str, indent=2))
        raise SystemExit(0 if report.get("passed") else 2)

    trace_id = new_trace_id("cli")

    if args.cmd == "halt":
        write_system_config(db, actor="cli", key="HALT_TRADING", value="true", trace_id=trace_id, reason_code="ADMIN_HALT", reason=args.reason)
        telegram.send_alert_zh(
            title="⏸️ 管理工具：暂停交易",
            summary_kv={"level": "WARN", "event": "ADMIN_HALT", "service": "管理工具", "trace_id": trace_id, "原因": args.reason},
            payload={"level": "WARN", "event": "ADMIN_HALT", "service": "admin-cli", "trace_id": trace_id, "reason_code": "ADMIN_HALT", "key": "HALT_TRADING", "value": "true", "reason": args.reason},
        )
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "resume":
        write_system_config(db, actor="cli", key="HALT_TRADING", value="false", trace_id=trace_id, reason_code="ADMIN_HALT", reason=args.reason)
        telegram.send_alert_zh(
            title="▶️ 管理工具：恢复交易",
            summary_kv={"level": "INFO", "event": "ADMIN_RESUME", "service": "管理工具", "trace_id": trace_id, "原因": args.reason},
            payload={"level": "INFO", "event": "ADMIN_RESUME", "service": "admin-cli", "trace_id": trace_id, "reason_code": "ADMIN_HALT", "key": "HALT_TRADING", "value": "false", "reason": args.reason},
        )
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "emergency-exit":
        write_system_config(db, actor="cli", key="EMERGENCY_EXIT", value="true", trace_id=trace_id, reason_code="EMERGENCY_EXIT", reason=args.reason)
        telegram.send_alert_zh(
            title="🆘 管理工具：紧急退出",
            summary_kv={"level": "CRITICAL", "event": "ADMIN_EMERGENCY_EXIT", "service": "管理工具", "trace_id": trace_id, "原因": args.reason},
            payload={"level": "CRITICAL", "event": "ADMIN_EMERGENCY_EXIT", "service": "admin-cli", "trace_id": trace_id, "reason_code": "EMERGENCY_EXIT", "key": "EMERGENCY_EXIT", "value": "true", "reason": args.reason},
        )
        print(f"OK trace_id={trace_id}")
        return


if __name__ == "__main__":
    main()
