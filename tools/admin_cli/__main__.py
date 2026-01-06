from __future__ import annotations

"""
alpha-admin（管理小工具）

你现在看到的 alpha-admin 只有：status / halt / resume / emergency-exit。
为了满足你“通过小工具一键测试所有流程”的需求，这个文件新增：

1) smoke-test：
   - 不下单
   - 一键检查：DB 连通 / Redis 连通 / 行情缓存是否更新（market_data_cache 是否足够新）
   - 适合上线前、改配置后快速验收

2) e2e-test（实盘闭环）：
   - 会真实下单（Binance / Bybit 二选一）
   - 流程：暂停策略(HALT_TRADING=true) -> 开仓(BUY) -> 平仓(SELL)
           -> 校验 SELL 的 pnl_usdt（交易所结算口径，含手续费影响）
   - 默认必须加 --yes 才会执行，避免误操作

运行方式（在宿主机项目目录）：
  docker compose exec api-service python -m tools.admin_cli smoke-test
  docker compose exec api-service python -m tools.admin_cli e2e-test --yes --qty 0.001
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional, Tuple

from shared.config import Settings
from shared.db import MariaDB
from shared.exchange import make_exchange
from shared.logging import new_trace_id
from shared.redis import redis_client
from shared.telemetry import Telegram
from shared.domain.events import append_order_event
from shared.domain.enums import OrderEventType, ReasonCode, Side


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
        INSERT INTO system_config(`key`, `value`) VALUES(%s,%s)
        ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)
        """,
        (key, value),
    )

    db.execute(
        """
        INSERT INTO config_audit(trace_id, actor, key_name, old_value, new_value, reason_code, reason)
        VALUES(%s,%s,%s,%s,%s,%s,%s)
        """,
        (trace_id, actor, key, old_val, value, reason_code, reason),
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

def _wait_for_market_cache(
    db: MariaDB,
    *,
    symbol: str,
    interval_minutes: int,
    wait_seconds: int,
    max_age_seconds: int,
) -> Tuple[bool, Dict[str, Any]]:
    """等待 market_data_cache 有最新数据。

    max_age_seconds：允许的“最新数据年龄”（例如 120 秒）。超过则判定数据不同步。
    """
    deadline = time.time() + wait_seconds
    last_row: Optional[Dict[str, Any]] = None

    while time.time() < deadline:
        row = db.fetch_one(
            """
            SELECT symbol, interval_minutes, open_time_ms, close_time_ms, close_price,
                   ema_fast, ema_slow, rsi_14
            FROM market_data_cache
            WHERE symbol=%s AND interval_minutes=%s
            ORDER BY open_time_ms DESC
            LIMIT 1
            """,
            (symbol, interval_minutes),
        )
        if row:
            last_row = dict(row)
            now_ms = int(time.time() * 1000)
            age_sec = int((now_ms - int(row["close_time_ms"])) / 1000)
            last_row["age_seconds"] = age_sec
            if age_sec <= max_age_seconds:
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

    # Telegram：中文文本 + JSON 摘要
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

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


# -----------------------------
# E2E Trade Test：实盘闭环（真实下单）
# -----------------------------

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


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

    # safety: 实盘必须显式确认
    if ex in ("binance", "bybit") and not yes:
        print(
            "[E2E] 该命令会真实下单（实盘/子账户）。为了避免误操作，必须加 --yes 才会执行。\n"
            "示例：docker compose exec api-service python -m tools.admin_cli e2e-test --yes --qty 0.001",
            file=sys.stderr,
        )
        return 2

    sym = (symbol or settings.symbol).upper()
    q = float(qty) if qty is not None else float(getattr(settings, "trade_qty", 0.0) or 0.0)
    if q <= 0:
        print("[E2E] qty 无效，请通过 --qty 指定一个满足交易所最小下单量的值。", file=sys.stderr)
        return 2

    db = MariaDB(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_pass,
        db=settings.db_name,
    )

    # 1) 先跑 smoke：保证 DB/Redis/行情缓存 OK
    smoke_rc = run_smoke_test(settings, wait_seconds=wait_seconds, max_age_seconds=max_age_seconds)
    if smoke_rc != 0:
        print("[E2E] smoke-test 未通过，终止 e2e-test。", file=sys.stderr)
        return 2

    # 2) 暂停策略引擎（避免策略同时下单影响测试）
    old_halt = read_system_config(db, "HALT_TRADING", "false")
    if ex != "paper":
        write_system_config(
            db,
            actor="cli",
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

    # 3) 下单：BUY -> SELL
    ex_client = make_exchange(settings, metrics=None, service_name="admin-cli")

    try:
        # BUY（开仓）
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

        try:
            append_order_event(
                db,
                trace_id=trace_id,
                service="admin-cli",
                exchange=settings.exchange,
                symbol=sym,
                client_order_id=client_buy,
                exchange_order_id=buy.exchange_order_id,
                event_type=OrderEventType.FILLED,
                side=Side.BUY.value,
                qty=buy.filled_qty or q,
                price=_safe_float(buy.avg_price),
                status=buy.status,
                reason_code=ReasonCode.SYSTEM,
                reason="E2E_TEST_BUY",
                payload=report["results"]["buy"],
            )
        except Exception:
            pass

        time.sleep(max(0.0, float(sleep_after_entry)))

        # SELL（平仓）
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

        try:
            append_order_event(
                db,
                trace_id=trace_id,
                service="admin-cli",
                exchange=settings.exchange,
                symbol=sym,
                client_order_id=client_sell,
                exchange_order_id=sell.exchange_order_id,
                event_type=OrderEventType.FILLED,
                side=Side.SELL.value,
                qty=sell.filled_qty or q,
                price=_safe_float(sell.avg_price),
                status=sell.status,
                reason_code=ReasonCode.SYSTEM,
                reason="E2E_TEST_SELL",
                payload=report["results"]["sell"],
            )
        except Exception:
            pass

        pnl = sell.pnl_usdt
        ok = pnl is not None

        # Telegram 告警（中文文本 + JSON 摘要）
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

        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if ok else 2

    except Exception as e:
        report["error"] = str(e)
        if telegram.enabled():
            telegram.send_alert_zh(
                title="❌ E2E 测试异常",
                summary_kv={"trace_id": trace_id, "错误": str(e)},
                payload=report,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    finally:
        # 4) 恢复 HALT_TRADING（避免影响你正常运行）
        if restore_halt:
            try:
                write_system_config(
                    db,
                    actor="cli",
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

    # status
    p_status = sub.add_parser("status", help="查看系统状态（DB/Redis/缓存/开关）")
    p_status.add_argument("--max-age-seconds", type=int, default=120, help="行情缓存允许的最大延迟（秒）")
    p_status.add_argument("--wait-seconds", type=int, default=30, help="等待行情缓存的最长时间（秒）")

    # halt/resume/emergency-exit
    p_halt = sub.add_parser("halt", help="暂停交易（写入 HALT_TRADING=true）")
    p_halt.add_argument("--reason", default="manual halt", help="原因（用于审计/告警）")

    p_resume = sub.add_parser("resume", help="恢复交易（写入 HALT_TRADING=false）")
    p_resume.add_argument("--reason", default="manual resume", help="原因（用于审计/告警）")

    p_exit = sub.add_parser("emergency-exit", help="紧急退出（写入 EMERGENCY_EXIT=true，供策略引擎执行）")
    p_exit.add_argument("--reason", default="manual emergency exit", help="原因（用于审计/告警）")

    # smoke-test
    p_smoke = sub.add_parser("smoke-test", help="一键链路自检（不下单）：DB/Redis/行情缓存")
    p_smoke.add_argument("--wait-seconds", type=int, default=120, help="等待行情缓存 ready 的最长时间（秒）")
    p_smoke.add_argument("--max-age-seconds", type=int, default=120, help="行情缓存允许的最大延迟（秒）")

    # e2e-test
    p_e2e = sub.add_parser("e2e-test", help="一键实盘闭环：开仓(BUY)->平仓(SELL)->校验真实 pnl_usdt（需 --yes）")
    p_e2e.add_argument("--yes", action="store_true", help="确认执行真实下单（Binance/Bybit）")
    p_e2e.add_argument("--qty", type=float, default=None, help="下单数量（覆盖 .env 的 TRADE_QTY）")
    p_e2e.add_argument("--symbol", type=str, default=None, help="交易对（覆盖 .env 的 SYMBOL）")
    p_e2e.add_argument("--wait-seconds", type=int, default=120, help="等待行情缓存 ready 的最长时间（秒）")
    p_e2e.add_argument("--max-age-seconds", type=int, default=120, help="行情缓存允许的最大延迟（秒）")
    p_e2e.add_argument("--sleep-after-entry", type=float, default=0.5, help="开仓后等待多少秒再平仓（避免交易所延迟）")
    p_e2e.add_argument("--no-restore-halt", action="store_true", help="不恢复原来的 HALT_TRADING（默认会恢复）")

    args = parser.parse_args()

    # 共享 DB 连接（status/halt/resume/emergency-exit 会用）
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
        report = {
            "env": getattr(settings, "env", getattr(settings, "app_env", "")),
            "exchange": settings.exchange,
            "symbol": settings.symbol,
            "interval_minutes": settings.interval_minutes,
            "db_ping": bool(db.ping()),
        }

        # Redis
        try:
            r = redis_client(settings.redis_url)
            report["redis_ping"] = bool(r.ping())
        except Exception as e:
            report["redis_ping"] = False
            report["redis_error"] = str(e)

        # flags
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

        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.cmd == "halt":
        write_system_config(
            db,
            actor="cli",
            key="HALT_TRADING",
            value="true",
            trace_id=trace_id,
            reason_code="ADMIN_HALT",
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
        write_system_config(
            db,
            actor="cli",
            key="HALT_TRADING",
            value="false",
            trace_id=trace_id,
            reason_code="ADMIN_HALT",
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
        write_system_config(
            db,
            actor="cli",
            key="EMERGENCY_EXIT",
            value="true",
            trace_id=trace_id,
            reason_code="EMERGENCY_EXIT",
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

    if args.cmd == "smoke-test":
        raise SystemExit(run_smoke_test(settings, wait_seconds=int(args.wait_seconds), max_age_seconds=int(args.max_age_seconds)))

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


if __name__ == "__main__":
    main()
