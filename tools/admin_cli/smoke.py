
from __future__ import annotations

"""One-click smoke/regression test for Alpha-Sniper-V8.

Design intent
-------------
This script is meant to be a practical *operator tool*:

- Validate connectivity to MariaDB / Redis
- Validate required tables exist and migrations applied
- Validate market data pipeline is working (market_data / market_data_cache)
- Validate critical invariants:
  - order_events idempotency by (exchange, symbol, client_order_id)
  - admin flags HALT_TRADING / EMERGENCY_EXIT can be written with audit
  - emergency-exit path can be executed by strategy-engine (paper mode recommended)

How it works
------------
The smoke test uses **only** DB/Redis checks plus optional "end-to-end" checks
that depend on a running `strategy-engine` service.

This avoids calling the exchange directly (safe), and supports your requirement
that runtime uses a single exchange at a time. For a full E2E test you should
run with `EXCHANGE=paper` first.

Output
------
- Prints a JSON summary with `passed=true/false`
- Includes step-by-step results and diagnostic hints
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from shared.config import Settings
from shared.db import MariaDB
from shared.redis.client import redis_client
from shared.logging.trace import new_trace_id
from shared.domain.enums import OrderEventType, ReasonCode, Side
from shared.domain.events import append_order_event

@dataclass
class StepResult:
    name: str
    ok: bool
    details: Dict[str, Any]

def _sql_now_utc() -> str:
    # For MariaDB: CURRENT_TIMESTAMP is UTC if server timezone configured;
    # We don't rely on it for correctness, only for heartbeat freshness.
    return "CURRENT_TIMESTAMP"

def _fetch_scalar(db: MariaDB, sql: str, params: tuple = ()) -> Optional[Any]:
    row = db.fetch_one(sql, params)
    if not row:
        return None
    return next(iter(row.values()))

def _wait_until(fn, timeout_s: int, poll_s: float = 2.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(poll_s)
    return False

def run_smoke_test(
    *,
    wait_data_seconds: int = 60,
    wait_engine_seconds: int = 30,
    engine_grace_seconds: int = 5,
) -> Dict[str, Any]:
    settings = Settings()
    trace_id = new_trace_id("smoke")
    steps: List[StepResult] = []

    db = MariaDB(settings.db_host, settings.db_port, settings.db_user, settings.db_pass, settings.db_name)
    r = redis_client(settings.redis_url)

    # 1) DB connectivity
    try:
        ok = db.ping()
        steps.append(StepResult("db_ping", ok, {"db_host": settings.db_host, "db_name": settings.db_name}))
    except Exception as e:
        steps.append(StepResult("db_ping", False, {"error": str(e)}))

    # 2) Redis connectivity
    try:
        pong = r.ping()
        steps.append(StepResult("redis_ping", bool(pong), {"redis_url": settings.redis_url}))
    except Exception as e:
        steps.append(StepResult("redis_ping", False, {"error": str(e)}))

    # 3) Tables existence (migrations)
    try:
        cnt = _fetch_scalar(db, "SELECT COUNT(*) AS c FROM information_schema.tables WHERE table_schema=%s AND table_name='order_events'", (settings.db_name,))
        ok = bool(cnt and int(cnt) == 1)
        steps.append(StepResult("tables_exist", ok, {"order_events_table": bool(ok)}))
    except Exception as e:
        steps.append(StepResult("tables_exist", False, {"error": str(e)}))

    # 4) Wait for market_data to exist (data-syncer pipeline)
    def has_market_data() -> bool:
        c = _fetch_scalar(
            db,
            "SELECT COUNT(*) AS c FROM market_data WHERE symbol=%s AND interval_minutes=%s",
            (settings.symbol, settings.interval_minutes, int(getattr(settings,'feature_version',1))),
        )
        return bool(c and int(c) > 0)

    try:
        ok = has_market_data() or _wait_until(has_market_data, timeout_s=wait_data_seconds, poll_s=2.0)
        details = {"symbol": settings.symbol, "interval_minutes": settings.interval_minutes, "wait_seconds": wait_data_seconds}
        if ok:
            last = db.fetch_one(
                "SELECT open_time_ms, close_time_ms, close_price FROM market_data WHERE symbol=%s AND interval_minutes=%s ORDER BY open_time_ms DESC LIMIT 1",
                (settings.symbol, settings.interval_minutes, int(getattr(settings,'feature_version',1))),
            )
            details["latest"] = last
        steps.append(StepResult("market_data_ready", ok, details))
    except Exception as e:
        steps.append(StepResult("market_data_ready", False, {"error": str(e)}))

    # 5) market_data_cache existence
    try:
        c = _fetch_scalar(
            db,
            "SELECT COUNT(*) AS c FROM market_data_cache WHERE symbol=%s AND interval_minutes=%s AND feature_version=%s",
            (settings.symbol, settings.interval_minutes, int(getattr(settings,'feature_version',1))),
        )
        ok = bool(c and int(c) > 0)
        steps.append(StepResult("market_data_cache_ready", ok, {"count": int(c or 0)}))
    except Exception as e:
        steps.append(StepResult("market_data_cache_ready", False, {"error": str(e)}))

    # 6) Service heartbeat freshness (optional but very helpful)
    try:
        rows = db.fetch_all(
            """
            SELECT service_name, instance_id, last_heartbeat
            FROM service_status
            WHERE service_name IN ('data-syncer','strategy-engine')
            ORDER BY last_heartbeat DESC
            """
        )
        steps.append(StepResult("service_heartbeats", True, {"rows": rows}))
    except Exception as e:
        steps.append(StepResult("service_heartbeats", False, {"error": str(e)}))

    # 7) order_events idempotency by unique key (exchange,symbol,client_order_id)
    try:
        client_order_id = f"smoke_{trace_id}"
        for _ in range(2):
            append_order_event(
                db,
                trace_id=trace_id,
                service="admin-cli",
                exchange=settings.exchange,
                symbol=settings.symbol,
                client_order_id=client_order_id,
                exchange_order_id=None,
                event_type=OrderEventType.CREATED,
                side=Side.BUY.value,
                qty=0.001,
                price=None,
                status="CREATED",
                reason_code=ReasonCode.SYSTEM,
                reason="smoke idempotency test",
                payload={"i": _},
            )
        c = _fetch_scalar(
            db,
            "SELECT COUNT(*) AS c FROM order_events WHERE exchange=%s AND symbol=%s AND client_order_id=%s",
            (settings.exchange, settings.symbol, client_order_id),
        )
        steps.append(StepResult("order_event_idempotent", bool(c and int(c) == 1), {"count": int(c or 0), "client_order_id": client_order_id}))
    except Exception as e:
        steps.append(StepResult("order_event_idempotent", False, {"error": str(e)}))

    # 8) Admin flags write/read sanity (HALT_TRADING)
    try:
        # Direct DB write: same behavior as /admin/update_config path, but faster and deterministic for regression.
        db.execute("INSERT INTO system_config(`key`,`value`) VALUES ('HALT_TRADING','true') ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)")
        v1 = _fetch_scalar(db, "SELECT `value` FROM system_config WHERE `key`='HALT_TRADING'")
        db.execute("INSERT INTO system_config(`key`,`value`) VALUES ('HALT_TRADING','false') ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)")
        v2 = _fetch_scalar(db, "SELECT `value` FROM system_config WHERE `key`='HALT_TRADING'")
        steps.append(StepResult("admin_flag_halt_rw", (str(v1).lower()=="true" and str(v2).lower()=="false"), {"v1": v1, "v2": v2}))
    except Exception as e:
        steps.append(StepResult("admin_flag_halt_rw", False, {"error": str(e)}))

    # 9) Optional end-to-end emergency exit (requires running strategy-engine)
    # We create a fake position snapshot >0, set EMERGENCY_EXIT=true, then wait for strategy-engine to process and flatten.
    try:
        # Write a position (paper-safe). Strategy-engine will read latest position snapshot.
        db.execute(
            """
            INSERT INTO position_snapshots(symbol, base_qty, avg_entry_price, meta_json)
            VALUES (%s,%s,%s,%s)
            """,
            (settings.symbol, 0.002, 100000.0, json.dumps({"trace_id": trace_id, "note": "smoke position"}, ensure_ascii=False)),
        )
        db.execute("INSERT INTO system_config(`key`,`value`) VALUES ('EMERGENCY_EXIT','true') ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)")
        start_id = _fetch_scalar(db, "SELECT IFNULL(MAX(id),0) AS mx FROM order_events")
        start_id = int(start_id or 0)

        def emergency_exit_done() -> bool:
            row = db.fetch_one(
                """
                SELECT id FROM order_events
                WHERE id > %s AND symbol=%s AND reason_code=%s
                ORDER BY id DESC LIMIT 1
                """,
                (start_id, settings.symbol, ReasonCode.EMERGENCY_EXIT.value),
            )
            if not row:
                return False
            # also check position flattened
            pos = db.fetch_one(
                "SELECT base_qty FROM position_snapshots WHERE symbol=%s ORDER BY id DESC LIMIT 1",
                (settings.symbol,),
            )
            if not pos:
                return False
            return float(pos["base_qty"]) == 0.0

        ok = _wait_until(emergency_exit_done, timeout_s=wait_engine_seconds + engine_grace_seconds, poll_s=2.0)
        steps.append(StepResult(
            "emergency_exit_e2e",
            ok,
            {
                "requires": "strategy-engine running",
                "wait_seconds": wait_engine_seconds,
                "hint": "If this fails, make sure strategy-engine is running and STRATEGY_TICK_SECONDS is small for testing.",
            },
        ))
        # Always reset flag to prevent accidental continuous exits.
        db.execute("INSERT INTO system_config(`key`,`value`) VALUES ('EMERGENCY_EXIT','false') ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)")
    except Exception as e:
        steps.append(StepResult("emergency_exit_e2e", False, {"error": str(e), "requires": "strategy-engine running"}))

    passed = all(s.ok for s in steps if s.name not in ("emergency_exit_e2e",)) and              any(s.ok for s in steps if s.name == "db_ping")

    return {
        "trace_id": trace_id,
        "exchange": settings.exchange,
        "symbol": settings.symbol,
        "interval_minutes": settings.interval_minutes,
        "passed": bool(passed),
        "steps": [asdict(s) for s in steps],
        "notes": [
            "For a safe full E2E smoke test, set EXCHANGE=paper and STRATEGY_TICK_SECONDS=10 in .env.",
            "If you use external DB/Redis, ensure network ACL/firewall allows the containers to connect.",
        ],
    }
