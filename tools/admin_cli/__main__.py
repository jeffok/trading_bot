from __future__ import annotations

"""
alpha-admin（管理小工具）

新增：
- smoke-test：不下单，一键检查 DB/Redis/行情缓存是否更新
- e2e-test：实盘闭环（BUY->SELL）并校验 SELL 的 pnl_usdt（交易所结算口径）

修复：
1) market_data_cache 表结构不一致：
   - 不强依赖 close_time_ms
   - SQL 用 SELECT * 避免 Unknown column
   - age_seconds 优先 close_time_ms；否则 open_time_ms + interval 推算

2) config_audit 字段名按现有表：
   - INSERT config_audit(actor, action, cfg_key, old_value, new_value, trace_id, reason_code, reason)

3) JSON 序列化：
   - report/payload 里可能有 Decimal / datetime
   - 所有 json.dumps 都带 default=_json_default
"""

import argparse
import datetime
import json
import sys
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.config import Settings
from shared.db import MariaDB
from shared.exchange import make_exchange
from shared.logging import new_trace_id
from shared.redis import redis_client
from shared.telemetry import Telegram


# -----------------------------
# JSON 序列化兜底（防 Decimal / datetime 崩溃）
# -----------------------------
def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        try:
            return float(o)
        except Exception:
            return str(o)
    return str(o)


# -----------------------------
# DB 工具：system_config 写入（带审计）
# -----------------------------

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
    """写 system_config，并记录 config_audit（用于审计/回溯）。"""
    old = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
    old_val = old["value"] if old else None

    db.execute(
        """
        INSERT INTO system_config(`key`, `value`)
        VALUES (%s, %s) ON DUPLICATE KEY
        UPDATE `value`=
        VALUES (`value`)
        """,
        (key, value),
    )

    # ✅ 匹配现有表结构
    db.execute(
        """
        INSERT INTO config_audit(actor, action, cfg_key, old_value, new_value, trace_id, reason_code, reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (actor, "SET", key, old_val, value, trace_id, reason_code, reason),
    )


def read_system_config(db: MariaDB, key: str, default: str = "") -> str:
    row = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (key,))
    if not row:
        return default
    v = row.get("value")
    return str(v) if v is not None else default


# -----------------------------
# Smoke Test：链路自检（不下单）
# -----------------------------

def expected_reason_code(got: str, expected: str) -> None:
    if got != expected:
        raise SystemExit(f"ERROR: --reason-code must be '{expected}' (got '{got}')")


def _dict_row(row: Any) -> Dict[str, Any]:
    try:
        return dict(row)
    except Exception:
        return {}


def require_confirm_cli(settings: Settings, confirm_code: str | None) -> None:
    if not getattr(settings, "admin_confirm_required", False):
        return
    if not getattr(settings, "admin_confirm_code", ""):
        raise SystemExit("ADMIN_CONFIRM_REQUIRED enabled but ADMIN_CONFIRM_CODE is empty")
    if not confirm_code or confirm_code != settings.admin_confirm_code:
        raise SystemExit("confirm_code required (ADMIN_CONFIRM_REQUIRED=true)")


def _calc_cache_age_seconds(row: Dict[str, Any], interval_minutes: int) -> Optional[int]:
    """
    计算 cache 最新记录的“年龄（秒）”
    - 优先 close_time_ms
    - 否则用 open_time_ms + interval 推算 close_time_ms
    """
    now_ms = int(time.time() * 1000)

    close_ms = row.get("close_time_ms")
    if close_ms is not None:
        try:
            return int((now_ms - int(close_ms)) / 1000)
        except Exception:
            pass

    open_ms = row.get("open_time_ms")
    if open_ms is None:
        return None
    try:
        close_ms2 = int(open_ms) + int(interval_minutes) * 60 * 1000
        return int((now_ms - close_ms2) / 1000)
    except Exception:
        return None


def _wait_for_market_cache(
        db: MariaDB,
        *,
        symbol: str,
        interval_minutes: int,
        wait_seconds: int,
        max_age_seconds: int,
) -> Tuple[bool, Dict[str, Any]]:
    """
    等待 market_data_cache 有最新数据。

    兼容不同表结构：
    - SELECT * 避免字段差异导致 1054
    - age_seconds 不强依赖 close_time_ms
    """
    deadline = time.time() + wait_seconds
    last_row: Optional[Dict[str, Any]] = None

    while time.time() < deadline:
        row = db.fetch_one(
            """
            SELECT *
            FROM market_data_cache
            WHERE symbol = %s
              AND interval_minutes = %s
            ORDER BY open_time_ms DESC LIMIT 1
            """,
            (symbol, interval_minutes),
        )

        if row:
            last_row = _dict_row(row)
            age_sec = _calc_cache_age_seconds(last_row, interval_minutes)
            last_row["age_seconds"] = age_sec

            if age_sec is not None and age_sec <= max_age_seconds:
                return True, last_row

        time.sleep(1.0)

    return False, (last_row or {})


def run_smoke_test(settings: Settings, *, wait_seconds: int, max_age_seconds: int) -> int:
    """执行链路自检。返回进程退出码：0=通过，2=失败。"""
    trace_id = new_trace_id("smoke")
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    report: Dict[str, Any] = {
        "trace_id": trace_id,
        "env": getattr(settings, "env", getattr(settings, "app_env", "")),
        "exchange": settings.exchange,
        "symbol": settings.symbol,
        "interval_minutes": settings.interval_minutes,
        "checks": {},
    }

    db = MariaDB(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_pass,
        db=settings.db_name,
    )

    # 1) DB
    try:
        report["checks"]["db_ping"] = bool(db.ping())
    except Exception as e:
        report["checks"]["db_ping"] = False
        report["checks"]["db_error"] = str(e)

    # 2) Redis
    try:
        r = redis_client(settings.redis_url)
        report["checks"]["redis_ping"] = bool(r.ping())
    except Exception as e:
        report["checks"]["redis_ping"] = False
        report["checks"]["redis_error"] = str(e)

    # 3) 行情缓存
    try:
        ok, last = _wait_for_market_cache(
            db,
            symbol=settings.symbol,
            interval_minutes=settings.interval_minutes,
            wait_seconds=wait_seconds,
            max_age_seconds=max_age_seconds,
        )
        report["checks"]["market_cache_ok"] = ok
        report["checks"]["market_cache_last"] = last
    except Exception as e:
        report["checks"]["market_cache_ok"] = False
        report["checks"]["market_cache_error"] = str(e)

    # 4) 管理开关（只读）
    try:
        report["checks"]["halt_trading"] = read_system_config(db, "HALT_TRADING", "false")
        report["checks"]["emergency_exit"] = read_system_config(db, "EMERGENCY_EXIT", "false")
    except Exception as e:
        report["checks"]["flags_error"] = str(e)

    passed = (
            report["checks"].get("db_ping") is True
            and report["checks"].get("redis_ping") is True
            and report["checks"].get("market_cache_ok") is True
    )

    # Telegram：中文文本 + JSON 摘要（send_alert_zh 内部已兜底 datetime/Decimal）
    if telegram.enabled():
        last = report["checks"].get("market_cache_last") or {}
        telegram.send_alert_zh(
            title="✅ Smoke Test 通过" if passed else "❌ Smoke Test 失败",
            summary_kv={
                "trace_id": trace_id,
                "交易所": settings.exchange,
                "交易对": settings.symbol,
                "DB": "OK" if report["checks"].get("db_ping") else "FAIL",
                "Redis": "OK" if report["checks"].get("redis_ping") else "FAIL",
                "行情缓存": "OK" if report["checks"].get("market_cache_ok") else "FAIL",
                "缓存延迟(秒)": last.get("age_seconds"),
            },
            payload=report,
        )

    # ✅ 修复：print 的 json.dumps 也要支持 Decimal/datetime
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if passed else 2


# -----------------------------
# E2E Trade Test：实盘闭环（真实下单）
# -----------------------------

def run_e2e_trade_test(
        settings: Settings,
        *,
        yes: bool,
        qty: Optional[float],
        symbol: Optional[str],
        wait_seconds: int,
        max_age_seconds: int,
        sleep_after_entry: float,
        restore_halt: bool,
) -> int:
    """实盘闭环测试：BUY -> SELL -> 校验 SELL 的 pnl_usdt（交易所结算口径，含手续费影响）。"""
    trace_id = new_trace_id("e2e")
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    ex = settings.exchange.lower()
    if ex not in ("binance", "bybit", "paper"):
        print(f"[E2E] 不支持的交易所 EXCHANGE={settings.exchange}", file=sys.stderr)
        return 2

    if ex in ("binance", "bybit") and not yes:
        print(
            "[E2E] 该命令会真实下单。为了避免误操作，必须加 --yes 才会执行。\n"
            "示例：docker compose exec api-service python -m tools.admin_cli e2e-test --yes --qty 0.001",
            file=sys.stderr,
        )
        return 2

    sym = (symbol or settings.symbol).upper()
    q = float(qty) if qty is not None else float(getattr(settings, "trade_qty", 0.0) or 0.0)
    if q <= 0:
        print("[E2E] qty 无效，请通过 --qty 指定一个满足交易所最小下单量的值。", file=sys.stderr)
        return 2

    # 1) 先跑 smoke：保证 DB/Redis/行情缓存 OK
    smoke_rc = run_smoke_test(settings, wait_seconds=wait_seconds, max_age_seconds=max_age_seconds)
    if smoke_rc != 0:
        print("[E2E] smoke-test 未通过，终止 e2e-test。", file=sys.stderr)
        return 2

    db = MariaDB(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_pass,
        db=settings.db_name,
    )

    # 2) 暂停策略引擎，避免策略同时下单影响测试
    old_halt = read_system_config(db, "HALT_TRADING", "false")
    if ex != "paper":
        write_system_config(
            db,
            actor=args.by,
            key="HALT_TRADING",
            value="true",
            trace_id=trace_id,
            reason_code="E2E_TEST",
            reason="e2e-test: pause strategy engine during test",
        )

    report: Dict[str, Any] = {
        "trace_id": trace_id,
        "exchange": settings.exchange,
        "symbol": sym,
        "qty": q,
        "results": {},
    }

    client_buy = f"e2e-buy-{trace_id}"
    client_sell = f"e2e-sell-{trace_id}"

    ex_client = make_exchange(settings, metrics=None, service_name="admin-cli")

    try:
        buy = ex_client.place_market_order(symbol=sym, side="BUY", qty=q, client_order_id=client_buy)
        report["results"]["buy"] = {
            "client_order_id": client_buy,
            "exchange_order_id": buy.exchange_order_id,
            "status": buy.status,
            "filled_qty": buy.filled_qty,
            "avg_price": buy.avg_price,
            "fee_usdt": buy.fee_usdt,
            "pnl_usdt": buy.pnl_usdt,
        }

        time.sleep(max(0.0, float(sleep_after_entry)))

        sell = ex_client.place_market_order(symbol=sym, side="SELL", qty=q, client_order_id=client_sell)
        report["results"]["sell"] = {
            "client_order_id": client_sell,
            "exchange_order_id": sell.exchange_order_id,
            "status": sell.status,
            "filled_qty": sell.filled_qty,
            "avg_price": sell.avg_price,
            "fee_usdt": sell.fee_usdt,
            "pnl_usdt": sell.pnl_usdt,
        }

        pnl = sell.pnl_usdt
        ok = pnl is not None

        if telegram.enabled():
            pnl_txt = "未知" if pnl is None else f"{pnl:.2f}"
            fee_txt = "未知" if sell.fee_usdt is None else f"{sell.fee_usdt:.2f}"
            telegram.send_alert_zh(
                title="✅ E2E 实盘闭环测试通过" if ok else "❌ E2E 实盘闭环测试失败",
                summary_kv={
                    "trace_id": trace_id,
                    "交易所": settings.exchange,
                    "交易对": sym,
                    "数量": q,
                    "平仓盈亏(USDT)": pnl_txt,
                    "手续费(USDT)": fee_txt,
                },
                payload=report,
            )

        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return 0 if ok else 2

    except Exception as e:
        report["error"] = str(e)
        if telegram.enabled():
            telegram.send_alert_zh(
                title="❌ E2E 测试异常",
                summary_kv={"trace_id": trace_id, "错误": str(e)},
                payload=report,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), file=sys.stderr)
        return 2

    finally:
        if restore_halt:
            try:
                write_system_config(
                    db,
                    actor=args.by,
                    key="HALT_TRADING",
                    value=str(old_halt),
                    trace_id=trace_id,
                    reason_code="E2E_TEST",
                    reason="e2e-test: restore HALT_TRADING",
                )
            except Exception:
                pass


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    settings = Settings()

    parser = argparse.ArgumentParser(prog="alpha-admin")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="查看系统状态（DB/Redis/缓存/开关）")
    p_status.add_argument("--max-age-seconds", type=int, default=120)
    p_status.add_argument("--wait-seconds", type=int, default=30)

    p_halt = sub.add_parser("halt", help="暂停交易（写入 HALT_TRADING=true）")
    p_halt.add_argument("--by", required=True, help="操作者/来源（写入审计 actor）")
    p_halt.add_argument("--reason-code", dest="reason_code", required=True, help="原因代码（建议 ADMIN_HALT）")
    p_halt.add_argument("--reason", required=True, help="原因说明")

    p_resume = sub.add_parser("resume", help="恢复交易（写入 HALT_TRADING=false）")
    p_resume.add_argument("--by", required=True, help="操作者/来源（写入审计 actor）")
    p_resume.add_argument("--reason-code", dest="reason_code", required=True, help="原因代码（建议 ADMIN_RESUME）")
    p_resume.add_argument("--reason", required=True, help="原因说明")

    p_exit = sub.add_parser("emergency-exit", help="紧急退出（写入 EMERGENCY_EXIT=true）")
    p_exit.add_argument("--by", required=True, help="操作者/来源（写入审计 actor）")
    p_exit.add_argument("--reason-code", dest="reason_code", required=True, help="原因代码（建议 EMERGENCY_EXIT）")
    p_exit.add_argument("--reason", required=True, help="原因说明")
    p_exit.add_argument("--confirm-code", dest="confirm_code", required=False,
                        help="二次确认码（若启用 ADMIN_CONFIRM_REQUIRED）")

    p_set = sub.add_parser("set", help="写入 system_config（等价于 /admin/update_config）")
    p_set.add_argument("key", type=str, help="配置键")
    p_set.add_argument("value", type=str, help="配置值")
    p_set.add_argument("--by", required=True, help="操作者/来源（写入审计 actor）")
    p_set.add_argument("--reason-code", dest="reason_code", required=True, help="原因代码（建议 ADMIN_UPDATE_CONFIG）")
    p_set.add_argument("--reason", required=True, help="原因说明")

    p_get = sub.add_parser("get", help="读取 system_config 的值")
    p_get.add_argument("key", type=str, help="配置键")

    p_list = sub.add_parser("list", help="列出 system_config（可选 prefix 过滤）")
    p_list.add_argument("--prefix", type=str, default="", help="key 前缀过滤")
    p_list.add_argument("--limit", type=int, default=200, help="最多返回条数")

    p_smoke = sub.add_parser("smoke-test", help="一键链路自检（不下单）：DB/Redis/行情缓存")
    p_smoke.add_argument("--wait-seconds", type=int, default=120)
    p_smoke.add_argument("--max-age-seconds", type=int, default=120)

    p_e2e = sub.add_parser("e2e-test", help="一键实盘闭环：BUY->SELL->校验真实 pnl_usdt（需 --yes）")
    p_e2e.add_argument("--yes", action="store_true")
    p_e2e.add_argument("--qty", type=float, default=None)
    p_e2e.add_argument("--symbol", type=str, default=None)
    p_e2e.add_argument("--wait-seconds", type=int, default=120)
    p_e2e.add_argument("--max-age-seconds", type=int, default=120)
    p_e2e.add_argument("--sleep-after-entry", type=float, default=0.5)
    p_e2e.add_argument("--no-restore-halt", action="store_true")

    args = parser.parse_args()

    if args.cmd == "set":
        expected_reason_code(args.reason_code, "ADMIN_UPDATE_CONFIG")
        require_confirm_cli(settings, getattr(args, "confirm_code", None))
        write_system_config(
            db,
            actor=args.by,
            key=args.key,
            value=args.value,
            trace_id=trace_id,
            reason_code=args.reason_code,
            reason=args.reason,
        )
        if telegram.enabled():
            telegram.send_alert_zh(
                title="⚙️ 已修改配置",
                summary_kv={"trace_id": trace_id, "key": args.key, "value": args.value, "原因": args.reason},
                payload={"trace_id": trace_id, "key": args.key, "value": args.value, "reason_code": args.reason_code,
                         "reason": args.reason},
            )
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "get":
        row = db.fetch_one("SELECT `value` FROM system_config WHERE `key`=%s", (args.key,))
        if not row:
            print("")
            return
        print(str(row["value"]))
        return

    if args.cmd == "list":
        prefix = (args.prefix or "").strip()
        limit = int(args.limit or 200)
        if prefix:
            rows = db.fetch_all(
                "SELECT `key`,`value`,updated_at FROM system_config WHERE `key` LIKE %s ORDER BY `key` ASC LIMIT %s",
                (prefix + "%", limit),
            )
        else:
            rows = db.fetch_all(
                "SELECT `key`,`value`,updated_at FROM system_config ORDER BY `key` ASC LIMIT %s",
                (limit,),
            )
        for r in rows or []:
            print(f"{r['key']}={r['value']}  (updated_at={r['updated_at']})")
        return
    if args.cmd == "smoke-test":
        raise SystemExit(
            run_smoke_test(settings, wait_seconds=int(args.wait_seconds), max_age_seconds=int(args.max_age_seconds)))

    if args.cmd == "e2e-test":
        raise SystemExit(
            run_e2e_trade_test(
                settings,
                yes=bool(args.yes),
                qty=args.qty,
                symbol=args.symbol,
                wait_seconds=int(args.wait_seconds),
                max_age_seconds=int(args.max_age_seconds),
                sleep_after_entry=float(args.sleep_after_entry),
                restore_halt=(not bool(args.no_restore_halt)),
            )
        )

    # 下面是原有简单命令
    db = MariaDB(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_pass,
        db=settings.db_name,
    )
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    trace_id = new_trace_id("admin")

    if args.cmd == "status":
        report: Dict[str, Any] = {
            "env": getattr(settings, "env", getattr(settings, "app_env", "")),
            "exchange": settings.exchange,
            "symbol": settings.symbol,
            "interval_minutes": settings.interval_minutes,
            "db_ping": bool(db.ping()),
        }
        try:
            r = redis_client(settings.redis_url)
            report["redis_ping"] = bool(r.ping())
        except Exception as e:
            report["redis_ping"] = False
            report["redis_error"] = str(e)

        report["halt_trading"] = read_system_config(db, "HALT_TRADING", "false")
        report["emergency_exit"] = read_system_config(db, "EMERGENCY_EXIT", "false")

        ok, last = _wait_for_market_cache(
            db,
            symbol=settings.symbol,
            interval_minutes=settings.interval_minutes,
            wait_seconds=int(args.wait_seconds),
            max_age_seconds=int(args.max_age_seconds),
        )
        report["market_cache_ok"] = ok
        report["market_cache_last"] = last

        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return

    if args.cmd == "halt":
        expected_reason_code(args.reason_code, "ADMIN_HALT")
        write_system_config(
            db,
            actor=args.by,
            key="HALT_TRADING",
            value="true",
            trace_id=trace_id,
            reason_code=args.reason_code,
            reason=args.reason,
        )
        if telegram.enabled():
            telegram.send_alert_zh(
                title="⏸️ 已暂停交易",
                summary_kv={"trace_id": trace_id, "原因": args.reason},
                payload={"trace_id": trace_id, "key": "HALT_TRADING", "value": "true", "reason": args.reason},
            )
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "resume":
        expected_reason_code(args.reason_code, "ADMIN_RESUME")
        write_system_config(
            db,
            actor=args.by,
            key="HALT_TRADING",
            value="false",
            trace_id=trace_id,
            reason_code=args.reason_code,
            reason=args.reason,
        )
        if telegram.enabled():
            telegram.send_alert_zh(
                title="▶️ 已恢复交易",
                summary_kv={"trace_id": trace_id, "原因": args.reason},
                payload={"trace_id": trace_id, "key": "HALT_TRADING", "value": "false", "reason": args.reason},
            )
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "emergency-exit":
        expected_reason_code(args.reason_code, "EMERGENCY_EXIT")
        require_confirm_cli(settings, getattr(args, "confirm_code", None))
        write_system_config(
            db,
            actor=args.by,
            key="EMERGENCY_EXIT",
            value="true",
            trace_id=trace_id,
            reason_code=args.reason_code,
            reason=args.reason,
        )
        if telegram.enabled():
            telegram.send_alert_zh(
                title="🧯 已触发紧急退出",
                summary_kv={"trace_id": trace_id, "原因": args.reason},
                payload={"trace_id": trace_id, "key": "EMERGENCY_EXIT", "value": "true", "reason": args.reason},
            )
        print(f"OK trace_id={trace_id}")
        return


if __name__ == "__main__":
    main()