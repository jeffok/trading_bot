"""测试不同Setup B条件组合的盈利情况

功能：
1. 测试单个条件
2. 测试条件组合（2个、3个、4个条件）
3. 计算每个组合的盈利指标（胜率、盈亏比、总盈利、最大回撤等）
4. 生成对比报告，推荐最佳组合

使用方式：
    python3 -m scripts.trading_test_tool.backtest_condition_combinations \
        --symbol BTCUSDT \
        --months 6 \
        --combinations "adx_di+squeeze_release,adx_di+momentum_flip,adx_di+volume_ratio"
"""

from __future__ import annotations

import json
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from shared.config import Settings, load_settings
from shared.db import PostgreSQL
from shared.logging import get_logger, new_trace_id
from shared.domain.runtime_config import RuntimeConfig
from services.strategy_engine.main import (
    setup_b_decision,
    compute_robot_score,
    leverage_from_score,
    min_qty_from_min_margin_usdt,
    enforce_risk_budget,
    compute_base_margin_usdt,
)

logger = get_logger("backtest_combinations", "INFO")


def _parse_json_maybe(s: object) -> dict:
    """解析JSONB字段，可能是字符串或字典"""
    try:
        if s is None:
            return {}
        if isinstance(s, dict):
            return s
        if isinstance(s, str):
            s2 = s.strip()
            if not s2:
                return {}
            return json.loads(s2)
        return {}
    except Exception:
        return {}


def check_condition(
    current: dict,
    prev: dict | None,
    *,
    condition_name: str,
    runtime_cfg: RuntimeConfig | None,
    settings: Settings,
    ai_score: float = 50.0,
) -> bool:
    """检查单个Setup B条件"""
    f = _parse_json_maybe(current.get("features_json"))
    fp = _parse_json_maybe(prev.get("features_json")) if prev else {}

    def _fnum(d: dict, k: str):
        try:
            v = d.get(k)
            return float(v) if v is not None else None
        except Exception:
            return None

    adx = _fnum(f, "adx14")
    pdi = _fnum(f, "plus_di14")
    mdi = _fnum(f, "minus_di14")
    vol_ratio = _fnum(f, "vol_ratio")
    mom = _fnum(f, "mom10")
    sq = _fnum(f, "squeeze_status")
    mom_prev = _fnum(fp, "mom10")
    sq_prev = _fnum(fp, "squeeze_status")

    if runtime_cfg:
        adx_min = float(runtime_cfg.setup_b_adx_min)
        vol_min = float(runtime_cfg.setup_b_vol_ratio_min)
        ai_min = float(runtime_cfg.setup_b_ai_score_min)
    else:
        adx_min = float(getattr(settings, "setup_b_adx_min", 20))
        vol_min = float(getattr(settings, "setup_b_vol_ratio_min", 1.5))
        ai_min = float(getattr(settings, "setup_b_ai_score_min", 55))

    squeeze_release = (sq_prev == 1.0 and sq == 0.0)
    mom_flip_pos = (mom_prev is not None and mom is not None and mom_prev < 0.0 and mom > 0.0)

    if condition_name == "adx_di":
        return adx is not None and adx >= adx_min and pdi is not None and mdi is not None and pdi > mdi
    elif condition_name == "squeeze_release":
        return squeeze_release
    elif condition_name == "momentum_flip":
        return mom_flip_pos
    elif condition_name == "volume_ratio":
        return vol_ratio is not None and vol_ratio >= vol_min
    elif condition_name == "ai_score":
        return float(ai_score) >= ai_min
    else:
        return False


def check_combination(
    current: dict,
    prev: dict | None,
    *,
    condition_names: List[str],
    runtime_cfg: RuntimeConfig | None,
    settings: Settings,
    ai_score: float = 50.0,
) -> bool:
    """检查条件组合（所有条件都必须满足）"""
    for cond_name in condition_names:
        if not check_condition(current, prev, condition_name=cond_name, runtime_cfg=runtime_cfg, settings=settings, ai_score=ai_score):
            return False
    return True


def simulate_trade(
    *,
    entry_price: float,
    entry_time_ms: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    leverage: float,
    base_margin_usdt: float,
    fee_rate: float,
    slippage_rate: float,
    klines: List[dict],
    start_idx: int,
) -> Dict[str, Any]:
    """模拟单笔交易"""
    # 计算实际入场价格（含滑点）
    entry_price_actual = entry_price * (1.0 + slippage_rate)
    
    # 计算止损和止盈价格
    stop_price = entry_price_actual * (1.0 - stop_loss_pct)
    take_profit_price = entry_price_actual * (1.0 + take_profit_pct)
    
    # 计算仓位数量
    qty = (base_margin_usdt * leverage) / entry_price_actual
    
    # 入场手续费
    entry_fee = base_margin_usdt * fee_rate
    
    # 查找出场点（止损或止盈）
    exit_idx = None
    exit_price = None
    exit_reason = None
    
    for i in range(start_idx + 1, len(klines)):
        kline = klines[i]
        high = float(kline.get("high_price", 0))
        low = float(kline.get("low_price", 0))
        
        # 检查是否触发止损
        if low <= stop_price:
            exit_idx = i
            exit_price = stop_price
            exit_reason = "stop_loss"
            break
        
        # 检查是否触发止盈
        if high >= take_profit_price:
            exit_idx = i
            exit_price = take_profit_price
            exit_reason = "take_profit"
            break
    
    # 如果没有触发止损或止盈，使用最后一根K线的收盘价
    if exit_idx is None:
        exit_idx = len(klines) - 1
        exit_price = float(klines[exit_idx].get("close_price", entry_price_actual))
        exit_reason = "timeout"
    
    # 计算出场价格（含滑点）
    if exit_reason == "stop_loss":
        exit_price_actual = exit_price * (1.0 - slippage_rate)
    else:
        exit_price_actual = exit_price * (1.0 + slippage_rate)
    
    # 出场手续费
    exit_fee = (qty * exit_price_actual) * fee_rate
    
    # 计算盈亏
    pnl = (exit_price_actual - entry_price_actual) * qty
    total_fee = entry_fee + exit_fee
    net_pnl = pnl - total_fee
    
    # 计算持仓时间（K线数量）
    holding_periods = exit_idx - start_idx
    
    return {
        "entry_price": entry_price_actual,
        "exit_price": exit_price_actual,
        "entry_time_ms": entry_time_ms,
        "exit_time_ms": int(klines[exit_idx].get("open_time_ms", entry_time_ms)),
        "qty": qty,
        "pnl": pnl,
        "total_fee": total_fee,
        "net_pnl": net_pnl,
        "exit_reason": exit_reason,
        "holding_periods": holding_periods,
        "leverage": leverage,
        "base_margin_usdt": base_margin_usdt,
    }


def backtest_combination(
    db: PostgreSQL,
    settings: Settings,
    runtime_cfg: RuntimeConfig,
    *,
    symbol: str,
    interval_minutes: int,
    feature_version: int,
    start_time_ms: int,
    end_time_ms: int,
    condition_names: List[str],
    initial_equity_usdt: float,
    fee_rate: float,
    slippage_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> Dict[str, Any]:
    """回测特定条件组合"""
    # 获取所有K线数据
    rows = db.fetch_all(
        """
        SELECT 
            mdc.open_time_ms,
            md.close_price,
            md.open_price,
            md.high_price,
            md.low_price,
            md.volume,
            mdc.ema_fast,
            mdc.ema_slow,
            mdc.rsi,
            mdc.features_json
        FROM market_data_cache mdc
        JOIN market_data md
          ON mdc.symbol = md.symbol
         AND mdc.interval_minutes = md.interval_minutes
         AND mdc.open_time_ms = md.open_time_ms
        WHERE mdc.symbol = %s
          AND mdc.interval_minutes = %s
          AND mdc.feature_version = %s
          AND mdc.open_time_ms >= %s
          AND mdc.open_time_ms <= %s
        ORDER BY mdc.open_time_ms ASC
        """,
        (symbol, interval_minutes, feature_version, start_time_ms, end_time_ms),
    )
    
    if not rows or len(rows) < 2:
        return {
            "combination": "+".join(condition_names),
            "signals": 0,
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_fee": 0.0,
            "net_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
        }
    
    # 查找所有信号
    signals = []
    for i in range(1, len(rows)):
        current = rows[i]
        prev = rows[i - 1]
        
        # 使用固定的AI评分（50.0），因为回测时无法实时计算AI评分
        ai_score = 50.0
        
        if check_combination(
            current,
            prev,
            condition_names=condition_names,
            runtime_cfg=runtime_cfg,
            settings=settings,
            ai_score=ai_score,
        ):
            signals.append(i)
    
    if not signals:
        return {
            "combination": "+".join(condition_names),
            "signals": 0,
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_fee": 0.0,
            "net_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
        }
    
    # 模拟交易（每次使用固定的初始资金，不使用滚仓）
    total_net_pnl = 0.0  # 累计净盈亏
    trades = []
    equity_curve = [initial_equity_usdt]
    current_position = None
    
    for signal_idx in signals:
        # 如果已有持仓，跳过
        if current_position is not None:
            continue
        
        kline = rows[signal_idx]
        entry_price = float(kline.get("close_price", 0))
        if entry_price <= 0:
            continue
        
        # 每次交易都使用固定的初始资金计算仓位（不使用滚仓）
        # 计算杠杆和保证金（基于初始资金）
        score = compute_robot_score(kline, signal="BUY")
        leverage = leverage_from_score(settings, score)
        base_margin_usdt = compute_base_margin_usdt(equity_usdt=initial_equity_usdt, ai_score=50.0, settings=settings)
        
        # 检查风险预算（基于初始资金）
        ok_risk, leverage_adj, _ = enforce_risk_budget(
            equity_usdt=initial_equity_usdt,
            base_margin_usdt=base_margin_usdt,
            leverage=leverage,
            stop_dist_pct=stop_loss_pct,
            settings=settings,
            runtime_cfg=runtime_cfg,
        )
        
        if not ok_risk:
            continue
        
        leverage = leverage_adj
        
        # 模拟交易
        trade = simulate_trade(
            entry_price=entry_price,
            entry_time_ms=int(kline.get("open_time_ms", 0)),
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            leverage=leverage,
            base_margin_usdt=base_margin_usdt,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            klines=rows,
            start_idx=signal_idx,
        )
        
        trades.append(trade)
        
        # 累计净盈亏（但不用于下次交易的资金计算）
        total_net_pnl += trade["net_pnl"]
        
        # 更新权益曲线（用于计算最大回撤）
        current_equity = initial_equity_usdt + total_net_pnl
        equity_curve.append(current_equity)
        
        # 如果亏损，清空持仓（模拟单笔交易）
        current_position = None
    
    # 计算最终权益（初始资金 + 累计净盈亏）
    final_equity = initial_equity_usdt + total_net_pnl
    
    # 计算统计指标
    if not trades:
        return {
            "combination": "+".join(condition_names),
            "signals": len(signals),
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_fee": 0.0,
            "net_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
        }
    
    winning_trades = [t for t in trades if t["net_pnl"] > 0]
    losing_trades = [t for t in trades if t["net_pnl"] <= 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0.0
    total_pnl = sum(t["pnl"] for t in trades)
    total_fee = sum(t["total_fee"] for t in trades)
    net_pnl = sum(t["net_pnl"] for t in trades)
    avg_pnl = net_pnl / len(trades) if trades else 0.0
    
    # 计算最大回撤（基于权益曲线）
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity_val in equity_curve:
        if equity_val > peak:
            peak = equity_val
        drawdown = (peak - equity_val) / peak if peak > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # 计算夏普比率（简化版，使用平均收益和标准差）
    returns = [t["net_pnl"] / initial_equity_usdt for t in trades]
    if len(returns) > 1:
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        sharpe_ratio = (avg_return / std_dev) if std_dev > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # 计算盈亏比
    total_profit = sum(t["net_pnl"] for t in winning_trades) if winning_trades else 0.0
    total_loss = abs(sum(t["net_pnl"] for t in losing_trades)) if losing_trades else 0.0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)
    
    return {
        "combination": "+".join(condition_names),
        "signals": len(signals),
        "trades": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_fee": total_fee,
        "net_pnl": net_pnl,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "final_equity": final_equity,
        "return_pct": (final_equity - initial_equity_usdt) / initial_equity_usdt * 100.0,
    }


def generate_comparison_report(results: List[Dict[str, Any]]) -> str:
    """生成对比报告"""
    lines = [
        "=" * 120,
        "Setup B 条件组合盈利对比报告",
        "=" * 120,
        "",
    ]
    
    # 按总盈利排序
    sorted_results = sorted(results, key=lambda x: x.get("net_pnl", 0.0), reverse=True)
    
    # 表头
    lines.extend([
        f"{'组合':<30} | {'信号数':>8} | {'交易数':>8} | {'胜率':>8} | {'总盈利':>12} | {'收益率':>10} | {'最大回撤':>10} | {'盈亏比':>10}",
        "-" * 120,
    ])
    
    # 数据行
    for r in sorted_results:
        combination = r.get("combination", "")
        signals = r.get("signals", 0)
        trades = r.get("trades", 0)
        win_rate = r.get("win_rate", 0.0) * 100.0
        net_pnl = r.get("net_pnl", 0.0)
        return_pct = r.get("return_pct", 0.0)
        max_drawdown = r.get("max_drawdown", 0.0) * 100.0
        profit_factor = r.get("profit_factor", 0.0)
        
        lines.append(
            f"{combination:<30} | {signals:>8} | {trades:>8} | {win_rate:>7.1f}% | ${net_pnl:>11.2f} | {return_pct:>9.2f}% | {max_drawdown:>9.2f}% | {profit_factor:>9.2f}"
        )
    
    lines.extend([
        "",
        "=" * 120,
        "",
        "推荐组合（按盈利排序）：",
        "",
    ])
    
    # 推荐前3名
    for i, r in enumerate(sorted_results[:3], 1):
        combination = r.get("combination", "")
        net_pnl = r.get("net_pnl", 0.0)
        win_rate = r.get("win_rate", 0.0) * 100.0
        return_pct = r.get("return_pct", 0.0)
        max_drawdown = r.get("max_drawdown", 0.0) * 100.0
        
        lines.extend([
            f"{i}. {combination}",
            f"   总盈利: ${net_pnl:.2f} | 收益率: {return_pct:.2f}% | 胜率: {win_rate:.1f}% | 最大回撤: {max_drawdown:.2f}%",
            "",
        ])
    
    return "\n".join(lines)


def run_combination_backtest(
    *,
    symbol: str = "BTCUSDT",
    months: int = 6,
    interval_minutes: Optional[int] = None,
    combinations: Optional[List[str]] = None,
    initial_equity_usdt: float = 1000.0,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.001,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> int:
    """运行条件组合回测"""
    settings = load_settings()
    db = PostgreSQL(settings.postgres_url)
    
    runtime_cfg = RuntimeConfig.load(db, settings)
    
    interval_minutes = interval_minutes or int(settings.interval_minutes or 15)
    feature_version = int(settings.feature_version or 1)
    
    # 计算时间范围
    now_ms = int(time.time() * 1000)
    start_time_ms = now_ms - (months * 30 * 24 * 60 * 60 * 1000)
    interval_ms = interval_minutes * 60 * 1000
    start_time_ms = (start_time_ms // interval_ms) * interval_ms
    end_time_ms = now_ms
    
    trace_id = new_trace_id("backtest_combinations")
    logger.info(f"开始条件组合回测: symbol={symbol}, interval={interval_minutes}, trace_id={trace_id}")
    logger.info(f"时间范围: {datetime.fromtimestamp(start_time_ms/1000)} ~ {datetime.fromtimestamp(end_time_ms/1000)}")
    
    # 确定要测试的组合
    all_conditions = ["adx_di", "squeeze_release", "momentum_flip", "volume_ratio", "ai_score"]
    
    if combinations is None:
        # 默认测试所有单个条件和一些常见组合
        combinations = []
        # 单个条件
        combinations.extend(all_conditions)
        # 2个条件的组合
        for i in range(len(all_conditions)):
            for j in range(i + 1, len(all_conditions)):
                combinations.append(f"{all_conditions[i]}+{all_conditions[j]}")
        # 3个条件的组合（只测试前几个）
        combinations.extend([
            "adx_di+squeeze_release+momentum_flip",
            "adx_di+momentum_flip+volume_ratio",
            "adx_di+squeeze_release+volume_ratio",
        ])
        # 4个条件的组合
        combinations.append("adx_di+squeeze_release+momentum_flip+volume_ratio")
        # 5个条件（完整Setup B）
        combinations.append("+".join(all_conditions))
    else:
        # 解析用户提供的组合
        parsed_combinations = []
        for combo_str in combinations:
            parsed_combinations.append([c.strip() for c in combo_str.split("+")])
        combinations = parsed_combinations
    
    logger.info(f"将测试 {len(combinations)} 个组合")
    
    # 回测每个组合
    results = []
    for i, combo in enumerate(combinations, 1):
        if isinstance(combo, str):
            condition_names = [c.strip() for c in combo.split("+")]
        else:
            condition_names = combo
        
        combo_str = "+".join(condition_names)
        logger.info(f"[{i}/{len(combinations)}] 测试组合: {combo_str}")
        
        result = backtest_combination(
            db,
            settings,
            runtime_cfg,
            symbol=symbol,
            interval_minutes=interval_minutes,
            feature_version=feature_version,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            condition_names=condition_names,
            initial_equity_usdt=initial_equity_usdt,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        
        results.append(result)
        logger.info(f"  结果: {result.get('trades', 0)} 笔交易, 总盈利: ${result.get('net_pnl', 0.0):.2f}")
    
    # 生成报告
    report = generate_comparison_report(results)
    sys.stdout.write(report)
    sys.stdout.flush()
    
    db.close()
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试Setup B条件组合的盈利情况")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--months", type=int, default=6, help="回测月数")
    parser.add_argument("--interval", type=int, default=None, help="K线周期（分钟）")
    parser.add_argument("--combinations", type=str, nargs="+", default=None, help="要测试的组合，用+连接，例如: adx_di+squeeze_release")
    parser.add_argument("--equity", type=float, default=1000.0, help="初始资金USDT")
    parser.add_argument("--fee-rate", type=float, default=0.0004, help="手续费率")
    parser.add_argument("--slippage-rate", type=float, default=0.001, help="滑点率")
    parser.add_argument("--stop-loss", type=float, default=0.02, help="止损比例（例如0.02表示2%）")
    parser.add_argument("--take-profit", type=float, default=0.04, help="止盈比例（例如0.04表示4%）")
    
    args = parser.parse_args()
    
    sys.exit(run_combination_backtest(
        symbol=args.symbol,
        months=args.months,
        interval_minutes=args.interval,
        combinations=args.combinations,
        initial_equity_usdt=args.equity,
        fee_rate=getattr(args, "fee_rate"),
        slippage_rate=getattr(args, "slippage_rate"),
        stop_loss_pct=getattr(args, "stop_loss"),
        take_profit_pct=getattr(args, "take_profit"),
    ))
