from __future__ import annotations

"""
Trading Test Tool - 交易系统管理工具（仅在Docker中使用）

⚠️ 重要：此工具只能在Docker容器中使用

使用方式：
    方式1: tbot <command> [args...]  (推荐，如果已安装)
    方式2: python -m scripts.trading_test_tool <command> [args...]
    方式3: ./scripts/tbot <command> [args...]

命令列表：
    - prepare: 准备检查（检查配置、服务状态等）
    - status: 查看系统状态
    - diagnose: 诊断为什么没有下单
    - check: 语法检查
    - halt: 暂停交易
    - resume: 恢复交易
    - emergency-exit: 紧急退出
    - set: 设置配置
    - get: 获取配置
    - list: 列出配置
    - smoke-test: 链路自检
    - e2e-test: 端到端测试
    - backtest: 历史回测工具（需要token）
    - query: SQL查询（调试用）
    - seed: 生成合成测试数据
    - restart: 重启服务
    - arm-stop: 启用保护止损订单

使用 --help 查看详细帮助：
    tbot --help
    tbot <command> --help
    
详细操作指南请查看: OPERATION_GUIDE.md（项目根目录）
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.config import Settings, load_settings
from shared.db import PostgreSQL
from shared.exchange import make_exchange
from shared.logging import new_trace_id
from shared.redis import redis_client
from shared.telemetry import Telegram, build_system_summary, send_system_alert, log_action
from shared.domain.control_commands import write_control_command


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
        db: PostgreSQL,
        *,
        actor: str,
        key: str,
        value: str,
        trace_id: str,
        reason_code: str,
        reason: str,
) -> None:
    """写 system_config，并记录 config_audit（用于审计/回溯）。"""
    old = db.fetch_one('SELECT "value" FROM system_config WHERE "key"=%s', (key,))
    old_val = old["value"] if old else None

    db.execute(
        """
        INSERT INTO system_config("key", "value")
        VALUES (%s, %s) ON CONFLICT ("key") DO UPDATE SET "value"=EXCLUDED."value"
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


def read_system_config(db: PostgreSQL, key: str, default: str = "") -> str:
    row = db.fetch_one('SELECT "value" FROM system_config WHERE "key"=%s', (key,))
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


def require_admin_token(settings: Settings, token: str | None) -> None:
    """验证管理员Token（CLI版本）"""
    # 如果未提供token，尝试从环境变量读取
    if not token:
        token = os.getenv("ADMIN_TOKEN", "").strip()
    
    if not token:
        raise SystemExit("ERROR: Admin token required. Use --token <token> or set ADMIN_TOKEN environment variable")
    
    # 检查token是否匹配
    if token != settings.admin_token:
        raise SystemExit("ERROR: Invalid admin token")


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
        db: PostgreSQL,
        *,
        symbol: str,
        interval_minutes: int,
        feature_version: int,
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
              AND feature_version = %s
            ORDER BY open_time_ms DESC LIMIT 1
            """,
            (symbol, interval_minutes, int(feature_version)),
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

    db = PostgreSQL(settings.postgres_url)

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
            feature_version=int(getattr(settings, 'feature_version', 1)),
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
        summary_kv = build_system_summary(
            event="SMOKE_TEST",
            trace_id=trace_id,
            level="INFO" if passed else "ERROR",
            actor=args.by,
            exchange=settings.exchange,
            extra={
                "symbol": settings.symbol,
                "db": "OK" if report["checks"].get("db_ping") else "FAIL",
                "redis": "OK" if report["checks"].get("redis_ping") else "FAIL",
                "market_cache_ok": bool(report["checks"].get("market_cache_ok")),
                "market_cache_age_s": report["checks"].get("market_cache_age_seconds"),
                "market_cache_last_open_ms": (last.get("open_time_ms") if isinstance(last, dict) else None),
            },
        )
        send_system_alert(
            telegram,
            title="✅ Smoke Test 通过" if passed else "❌ Smoke Test 失败",
            summary_kv=summary_kv,
            payload={"report": report},
        )
        log_action(
            logger,
            action="SMOKE_TEST",
            trace_id=trace_id,
            reason_code="PASS" if passed else "FAIL",
            reason="smoke test passed" if passed else "smoke test failed",
            client_order_id=None,
            extra={"checks": report.get("checks")},
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
            "示例：docker compose exec execution python -m scripts.trading_test_tool e2e-test --yes --qty 0.001",
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

    db = PostgreSQL(settings.postgres_url)

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
            summary_kv = build_system_summary(
                event="E2E_TRADE_TEST",
                trace_id=trace_id,
                level="INFO" if ok else "ERROR",
                actor=args.by,
                exchange=settings.exchange,
                extra={"symbol": sym, "qty": q, "pnl_usdt": pnl_txt, "fee_usdt": fee_txt, "ok": bool(ok)},
            )
            send_system_alert(
                telegram,
                title="✅ E2E 实盘闭环测试通过" if ok else "❌ E2E 实盘闭环测试失败",
                summary_kv=summary_kv,
                payload={"report": report},
            )
            log_action(logger, action="E2E_TRADE_TEST", trace_id=trace_id, reason_code="PASS" if ok else "FAIL",
                       reason="e2e ok" if ok else "e2e failed", client_order_id=None, extra={"symbol": sym})

        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return 0 if ok else 2

    except Exception as e:
        report["error"] = str(e)
        if telegram.enabled():
            summary_kv = build_system_summary(
                event="E2E_TRADE_TEST_EXCEPTION",
                trace_id=trace_id,
                level="ERROR",
                actor=args.by,
                exchange=settings.exchange,
                reason=str(e),
            )
            send_system_alert(
                telegram,
                title="❌ E2E 测试异常",
                summary_kv=summary_kv,
                payload={"report": report, "error": str(e)},
            )
            log_action(logger, action="E2E_TRADE_TEST_EXCEPTION", trace_id=trace_id, reason_code="ERROR",
                       reason=str(e)[:200], client_order_id=None)
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
    settings = load_settings()

    parser = argparse.ArgumentParser(
        prog="tbot",
        description="交易系统管理工具（仅在Docker中使用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  tbot status                          # 查看系统状态
  tbot diagnose --symbol BTCUSDT      # 诊断指定交易对
  tbot backtest --token YOUR_TOKEN    # 历史回测
  tbot resume --by admin --reason-code ADMIN_RESUME --reason "恢复交易"
  
更多信息请查看项目根目录的 OPERATION_GUIDE.md
        """
    )
    sub = parser.add_subparsers(dest="cmd", required=True, help="可用命令")

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

    p_diagnose = sub.add_parser("diagnose", help="诊断为什么没有下单")
    p_diagnose.add_argument("--symbol", type=str, default=None, help="指定交易对（可选，默认诊断所有交易对）")

    p_check = sub.add_parser("check", help="语法检查（compileall）")

    p_query = sub.add_parser("query", help="执行SQL查询（仅用于调试）")
    p_query.add_argument("--sql", type=str, required=True, help="SQL查询语句")

    p_backtest = sub.add_parser("backtest", help="历史回测工具：分析Setup B信号出现次数")
    p_backtest.add_argument("--token", type=str, default=None, help="管理员Token（默认从 ADMIN_TOKEN 环境变量读取）")
    p_backtest.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对符号（默认：BTCUSDT）")
    p_backtest.add_argument("--months", type=int, default=6, help="回测月数（默认：6个月）")
    p_backtest.add_argument("--interval", type=int, default=None, help="K线周期（分钟，默认使用配置）")
    p_backtest.add_argument("--feature-version", type=int, default=None, dest="feature_version", help="特征版本（默认使用配置）")

    p_seed = sub.add_parser("seed", help="生成合成市场数据（用于测试）")
    p_seed.add_argument("--bars", type=int, default=260, help="生成的K线数量（默认：260）")
    p_seed.add_argument("--start-price", type=float, default=40000, dest="start_price", help="起始价格（默认：40000）")

    p_restart = sub.add_parser("restart", help="重启服务")
    p_restart.add_argument("service", type=str, choices=["data-syncer", "strategy-engine", "api-service", "all"], help="要重启的服务")

    p_arm_stop = sub.add_parser("arm-stop", help="启用保护止损订单")
    p_arm_stop.add_argument("--by", required=True, help="操作者/来源（写入审计 actor）")
    p_arm_stop.add_argument("--reason-code", dest="reason_code", required=True, help="原因代码（建议 ADMIN_UPDATE_CONFIG）")
    p_arm_stop.add_argument("--reason", required=True, help="原因说明")
    p_arm_stop.add_argument("--stop-poll-seconds", type=int, default=10, dest="stop_poll_seconds", help="止损单轮询间隔（默认：10秒）")

    p_config = sub.add_parser("config", help="输出所有配置参数（以JSON格式）")

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
        write_control_command(
            db,
            command="UPDATE_CONFIG",
            payload={"key": args.key, "value": args.value, "actor": args.by, "reason_code": args.reason_code,
                     "reason": args.reason, "trace_id": trace_id},
        )
        if telegram.enabled():
            summary_kv = build_system_summary(
                event="UPDATE_CONFIG",
                trace_id=trace_id,
                level="INFO",
                actor=args.by,
                reason_code=args.reason_code,
                reason=args.reason,
                extra={"key": args.key, "value": args.value},
            )
            send_system_alert(
                telegram,
                title="⚙️ 已修改配置",
                summary_kv=summary_kv,
                payload={"key": args.key, "value": args.value, "reason_code": args.reason_code, "reason": args.reason},
            )
            log_action(logger, action="UPDATE_CONFIG", trace_id=trace_id, reason_code=args.reason_code,
                       reason=args.reason, client_order_id=None, extra={"key": args.key})
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "get":
        row = db.fetch_one('SELECT "value" FROM system_config WHERE "key"=%s', (args.key,))
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
                'SELECT "key","value",updated_at FROM system_config WHERE "key" LIKE %s ORDER BY "key" ASC LIMIT %s',
                (prefix + "%", limit),
            )
        else:
            rows = db.fetch_all(
                'SELECT "key","value",updated_at FROM system_config ORDER BY "key" ASC LIMIT %s',
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

    if args.cmd == "diagnose":
        from scripts.trading_test_tool.diagnose import run_diagnose
        raise SystemExit(run_diagnose(settings, symbol=getattr(args, "symbol", None)))

    if args.cmd == "check":
        import compileall
        import os
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        ok = compileall.compile_dir(root, quiet=1)
        print("✅ 语法检查通过" if ok else "❌ 语法检查失败")
        raise SystemExit(0 if ok else 1)

    if args.cmd == "query":
        db = PostgreSQL(settings.postgres_url)
        try:
            rows = db.fetch_all(getattr(args, "sql", ""))
            print(json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default))
            return 0
        except Exception as e:
            print(f"❌ 查询失败: {e}", file=sys.stderr)
            return 1
        finally:
            db.close()

    if args.cmd == "backtest":
        require_admin_token(settings, getattr(args, "token", None))
        from scripts.trading_test_tool.backtest import run_backtest
        raise SystemExit(run_backtest(
            symbol=getattr(args, "symbol", "BTCUSDT"),
            months=getattr(args, "months", 6),
            interval_minutes=getattr(args, "interval", None),
            feature_version=getattr(args, "feature_version", None),
        ))

    if args.cmd == "seed":
        from scripts.trading_test_tool.seed import run_seed
        raise SystemExit(run_seed(
            bars=getattr(args, "bars", 260),
            start_price=getattr(args, "start_price", 40000),
        ))

    if args.cmd == "restart":
        service = getattr(args, "service", "all")
        try:
            if service == "all":
                subprocess.run(["docker", "compose", "restart", "data-syncer", "strategy-engine", "api-service"], check=True)
                print("✅ 已重启所有服务")
            else:
                subprocess.run(["docker", "compose", "restart", service], check=True)
                print(f"✅ 已重启服务: {service}")
            time.sleep(5)
            print("✅ 服务重启完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 重启服务失败: {e}", file=sys.stderr)
            raise SystemExit(1)
        except FileNotFoundError:
            print("❌ 错误: 未找到 docker compose 命令，请确保 Docker Compose 已安装", file=sys.stderr)
            raise SystemExit(1)

    # 下面是原有简单命令（需要 db 和 telegram）
    db = PostgreSQL(settings.postgres_url)
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    trace_id = new_trace_id("admin")

    if args.cmd == "prepare":
        # prepare命令等同于status，用于Docker环境
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
            feature_version=int(getattr(settings, 'feature_version', 1)),
            wait_seconds=int(args.wait_seconds),
            max_age_seconds=int(args.max_age_seconds),
        )
        report["market_cache_ok"] = ok
        report["market_cache_last"] = last

        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return

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
            feature_version=int(getattr(settings, 'feature_version', 1)),
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
        write_control_command(
            db,
            command="HALT",
            payload={"actor": args.by, "reason_code": args.reason_code, "reason": args.reason, "trace_id": trace_id},
        )
        if telegram.enabled():
            summary_kv = build_system_summary(
                event="HALT",
                trace_id=trace_id,
                level="WARN",
                actor=args.by,
                reason_code=args.reason_code,
                reason=args.reason,
            )
            send_system_alert(
                telegram,
                title="⏸️ 已暂停交易",
                summary_kv=summary_kv,
                payload={"key": "HALT_TRADING", "value": "true", "reason_code": args.reason_code,
                         "reason": args.reason},
            )
            log_action(logger, action="HALT", trace_id=trace_id, reason_code=args.reason_code, reason=args.reason,
                       client_order_id=None)
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
        write_control_command(
            db,
            command="RESUME",
            payload={"actor": args.by, "reason_code": args.reason_code, "reason": args.reason, "trace_id": trace_id},
        )
        if telegram.enabled():
            summary_kv = build_system_summary(
                event="RESUME",
                trace_id=trace_id,
                level="INFO",
                actor=args.by,
                reason_code=args.reason_code,
                reason=args.reason,
            )
            send_system_alert(
                telegram,
                title="▶️ 已恢复交易",
                summary_kv=summary_kv,
                payload={"key": "HALT_TRADING", "value": "false", "reason_code": args.reason_code,
                         "reason": args.reason},
            )
            log_action(logger, action="RESUME", trace_id=trace_id, reason_code=args.reason_code, reason=args.reason,
                       client_order_id=None)
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
        write_control_command(
            db,
            command="EMERGENCY_EXIT",
            payload={"actor": args.by, "reason_code": args.reason_code, "reason": args.reason, "trace_id": trace_id},
        )
        if telegram.enabled():
            summary_kv = build_system_summary(
                event="EMERGENCY_EXIT",
                trace_id=trace_id,
                level="WARN",
                actor=args.by,
                reason_code=args.reason_code,
                reason=args.reason,
            )
            send_system_alert(
                telegram,
                title="🧯 已触发紧急退出",
                summary_kv=summary_kv,
                payload={"key": "EMERGENCY_EXIT", "value": "true", "reason_code": args.reason_code,
                         "reason": args.reason},
            )
            log_action(logger, action="EMERGENCY_EXIT", trace_id=trace_id, reason_code=args.reason_code,
                       reason=args.reason, client_order_id=None)
        print(f"OK trace_id={trace_id}")
        return

    if args.cmd == "config":
        # 输出所有 Settings 配置参数
        import dataclasses
        config_dict = {}
        for field in dataclasses.fields(settings):
            value = getattr(settings, field.name, None)
            # 过滤敏感信息
            if field.name in ("admin_token", "binance_api_secret", "bybit_api_secret", "postgres_url", "redis_url"):
                config_dict[field.name] = "***REDACTED***"
            elif isinstance(value, tuple):
                config_dict[field.name] = list(value)
            else:
                config_dict[field.name] = value
        print(json.dumps(config_dict, ensure_ascii=False, indent=2, default=_json_default))
        return

    if args.cmd == "arm-stop":
        expected_reason_code(args.reason_code, "ADMIN_UPDATE_CONFIG")
        write_system_config(
            db,
            actor=args.by,
            key="USE_PROTECTIVE_STOP_ORDER",
            value="true",
            trace_id=trace_id,
            reason_code=args.reason_code,
            reason=args.reason,
        )
        write_system_config(
            db,
            actor=args.by,
            key="STOP_ORDER_POLL_SECONDS",
            value=str(getattr(args, "stop_poll_seconds", 10)),
            trace_id=trace_id,
            reason_code=args.reason_code,
            reason=args.reason,
        )
        if telegram.enabled():
            summary_kv = build_system_summary(
                event="ARM_STOP_ORDER",
                trace_id=trace_id,
                level="INFO",
                actor=args.by,
                reason_code=args.reason_code,
                reason=args.reason,
            )
            send_system_alert(
                telegram,
                title="🛡️ 已启用保护止损",
                summary_kv=summary_kv,
                payload={"USE_PROTECTIVE_STOP_ORDER": "true", "STOP_ORDER_POLL_SECONDS": str(getattr(args, "stop_poll_seconds", 10))},
            )
        print(f"OK trace_id={trace_id}")
        return


if __name__ == "__main__":
    main()