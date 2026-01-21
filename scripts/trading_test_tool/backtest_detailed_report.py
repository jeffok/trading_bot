"""生成详细回测报告：包含每次信号的完整信息

功能：
1. 测试指定币种和条件组合
2. 记录每次信号的详细信息（买卖时间、价格、收益率、盈亏比等）
3. 生成详细的交易报告

使用方式:
    python3 -m scripts.trading_test_tool.backtest_detailed_report \
        --symbol BTCUSDT \
        --combinations "adx_di+volume_ratio" \
        --months 6
"""

from __future__ import annotations

import json
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from shared.config import Settings, load_settings
from shared.db import PostgreSQL
from shared.logging import get_logger, new_trace_id
from shared.domain.runtime_config import RuntimeConfig
from services.strategy_engine.main import (
    compute_robot_score,
    leverage_from_score,
    min_qty_from_min_margin_usdt,
    enforce_risk_budget,
    compute_base_margin_usdt,
    _load_ai_model,
    _vectorize_for_ai,
)

logger = get_logger("backtest_detailed", "INFO")


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
    direction: str = "LONG",  # "LONG" 或 "SHORT"
) -> bool:
    """检查条件组合（所有条件都必须满足）
    
    Args:
        direction: "LONG" 做多（买涨）或 "SHORT" 做空（买跌）
    """
    f = _parse_json_maybe(current.get("features_json"))
    fp = _parse_json_maybe(prev.get("features_json")) if prev else {}

    def _fnum(d: dict, k: str):
        try:
            v = d.get(k)
            return float(v) if v is not None else None
        except Exception:
            return None

    # 对于做空，需要检查相反的条件
    if direction == "SHORT":
        # 做空条件：-DI > +DI（下降趋势）
        adx = _fnum(f, "adx14")
        pdi = _fnum(f, "plus_di14")
        mdi = _fnum(f, "minus_di14")
        mom = _fnum(f, "mom10")
        mom_prev = _fnum(fp, "mom10")
        
        if runtime_cfg:
            adx_min = float(runtime_cfg.setup_b_adx_min)
        else:
            adx_min = float(getattr(settings, "setup_b_adx_min", 20))
        
        # 做空条件：ADX >= 阈值 且 -DI > +DI（下降趋势）
        adx_ok = adx is not None and adx >= adx_min and mdi is not None and pdi is not None and mdi > pdi
        
        # 动量转负（从正转负）
        mom_flip_neg = (mom_prev is not None and mom is not None and mom_prev > 0.0 and mom < 0.0)
        
        # 检查其他条件（volume_ratio, squeeze_release 等对做空也适用）
        for cond_name in condition_names:
            if cond_name == "adx_di":
                if not adx_ok:
                    return False
            elif cond_name == "momentum_flip":
                if not mom_flip_neg:
                    return False
            else:
                # 其他条件（volume_ratio, squeeze_release）对做空也适用
                if not check_condition(current, prev, condition_name=cond_name, runtime_cfg=runtime_cfg, settings=settings, ai_score=ai_score):
                    return False
        
        return True
    else:
        # 做多（原有逻辑）
        for cond_name in condition_names:
            if not check_condition(current, prev, condition_name=cond_name, runtime_cfg=runtime_cfg, settings=settings, ai_score=ai_score):
                return False
        return True


def simulate_trade_detailed(
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
    direction: str = "LONG",  # "LONG" 做多 或 "SHORT" 做空
) -> Dict[str, Any]:
    """模拟单笔交易（详细版）
    
    Args:
        direction: "LONG" 做多（买涨）或 "SHORT" 做空（买跌）
    """
    # 计算实际入场价格（含滑点）
    if direction == "SHORT":
        # 做空：入场价减去滑点（卖出价）
        entry_price_actual = entry_price * (1.0 - slippage_rate)
        # 做空：止损价高于入场价，止盈价低于入场价
        stop_price = entry_price_actual * (1.0 + stop_loss_pct)
        take_profit_price = entry_price_actual * (1.0 - take_profit_pct)
    else:
        # 做多：入场价加上滑点（买入价）
        entry_price_actual = entry_price * (1.0 + slippage_rate)
        # 做多：止损价低于入场价，止盈价高于入场价
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
    exit_time_ms = None
    
    # 记录价格变化轨迹（用于计算最大浮盈/浮亏）
    price_trajectory = []
    
    for i in range(start_idx + 1, len(klines)):
        kline = klines[i]
        high = float(kline.get("high_price", 0))
        low = float(kline.get("low_price", 0))
        close = float(kline.get("close_price", entry_price_actual))
        kline_time_ms = int(kline.get("open_time_ms", entry_time_ms))
        
        # 记录价格轨迹
        price_trajectory.append({
            "time_ms": kline_time_ms,
            "high": high,
            "low": low,
            "close": close,
        })
        
        # 检查是否触发止损和止盈（根据方向不同）
        if direction == "SHORT":
            # 做空：价格上涨触发止损，价格下跌触发止盈
            if high >= stop_price:
                exit_idx = i
                exit_price = stop_price
                exit_reason = "stop_loss"
                exit_time_ms = kline_time_ms
                break
            if low <= take_profit_price:
                exit_idx = i
                exit_price = take_profit_price
                exit_reason = "take_profit"
                exit_time_ms = kline_time_ms
                break
        else:
            # 做多：价格下跌触发止损，价格上涨触发止盈
            if low <= stop_price:
                exit_idx = i
                exit_price = stop_price
                exit_reason = "stop_loss"
                exit_time_ms = kline_time_ms
                break
            if high >= take_profit_price:
                exit_idx = i
                exit_price = take_profit_price
                exit_reason = "take_profit"
                exit_time_ms = kline_time_ms
                break
    
    # 如果没有触发止损或止盈，使用最后一根K线的收盘价
    if exit_idx is None:
        exit_idx = len(klines) - 1
        last_kline = klines[exit_idx]
        exit_price = float(last_kline.get("close_price", entry_price_actual))
        exit_reason = "timeout"
        exit_time_ms = int(last_kline.get("open_time_ms", entry_time_ms))
    
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
    
    # 计算收益率
    return_pct = (net_pnl / base_margin_usdt) * 100.0 if base_margin_usdt > 0 else 0.0
    
    # 计算持仓时间（K线数量和时间）
    holding_periods = exit_idx - start_idx
    holding_hours = holding_periods * 0.25  # 15分钟K线 = 0.25小时
    holding_days = holding_hours / 24.0
    
    # 计算最大浮盈和最大浮亏
    max_unrealized_profit = 0.0
    max_unrealized_loss = 0.0
    
    for price_point in price_trajectory:
        if direction == "SHORT":
            # 做空：价格越低浮盈越大，价格越高浮亏越大
            unrealized_profit_low = (entry_price_actual - price_point["low"]) * qty - entry_fee
            if unrealized_profit_low > max_unrealized_profit:
                max_unrealized_profit = unrealized_profit_low
            unrealized_loss_high = (entry_price_actual - price_point["high"]) * qty - entry_fee
            if unrealized_loss_high < max_unrealized_loss:
                max_unrealized_loss = unrealized_loss_high
        else:
            # 做多：价格越高浮盈越大，价格越低浮亏越大
            unrealized_profit_high = (price_point["high"] - entry_price_actual) * qty - entry_fee
            if unrealized_profit_high > max_unrealized_profit:
                max_unrealized_profit = unrealized_profit_high
            unrealized_loss_low = (price_point["low"] - entry_price_actual) * qty - entry_fee
            if unrealized_loss_low < max_unrealized_loss:
                max_unrealized_loss = unrealized_loss_low
    
    # 计算盈亏比（相对于止损）
    risk_amount = base_margin_usdt * stop_loss_pct  # 风险金额
    reward_amount = base_margin_usdt * take_profit_pct  # 潜在收益金额
    risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0
    
    # 实际盈亏比（实际盈利/实际亏损）
    actual_profit = net_pnl if net_pnl > 0 else 0.0
    actual_loss = abs(net_pnl) if net_pnl < 0 else 0.0
    actual_risk_reward = actual_profit / actual_loss if actual_loss > 0 else (float("inf") if actual_profit > 0 else 0.0)
    
    return {
        "entry_price": entry_price,
        "entry_price_actual": entry_price_actual,
        "exit_price": exit_price,
        "exit_price_actual": exit_price_actual,
        "entry_time_ms": entry_time_ms,
        "exit_time_ms": exit_time_ms,
        "entry_time_str": datetime.fromtimestamp(entry_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_str": datetime.fromtimestamp(exit_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S"),
        "qty": qty,
        "leverage": leverage,
        "base_margin_usdt": base_margin_usdt,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "pnl": pnl,
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "total_fee": total_fee,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "exit_reason": exit_reason,
        "holding_periods": holding_periods,
        "holding_hours": holding_hours,
        "holding_days": holding_days,
        "max_unrealized_profit": max_unrealized_profit,
        "max_unrealized_loss": max_unrealized_loss,
        "risk_reward_ratio": risk_reward_ratio,
        "actual_risk_reward": actual_risk_reward,
        "risk_amount": risk_amount,
        "reward_amount": reward_amount,
        "direction": direction,
    }


def generate_detailed_report(
    trades: List[Dict[str, Any]],
    *,
    symbol: str,
    combination: str,
    initial_equity: float,
    final_equity: float,
    total_withdrawn: float = 0.0,
    withdrawal_count: int = 0,
    withdrawals: List[Dict[str, Any]] = None,
    total_profit: float = 0.0,
    ai_model_seen: int = 0,
) -> str:
    """生成详细报告"""
    lines = []
    
    lines.append("=" * 150)
    lines.append(f"详细回测报告 - {symbol} - {combination}")
    lines.append("=" * 150)
    lines.append("")
    
    # 总体统计
    total_trades = len(trades)
    winning_trades = [t for t in trades if t["net_pnl"] > 0]
    losing_trades = [t for t in trades if t["net_pnl"] <= 0]
    
    win_rate = len(winning_trades) / total_trades * 100.0 if total_trades > 0 else 0.0
    total_pnl = sum(t["pnl"] for t in trades)
    total_fee = sum(t["total_fee"] for t in trades)
    total_net_pnl = sum(t["net_pnl"] for t in trades)
    
    # 计算总收益（包括提取的金额）
    total_profit = total_profit if total_profit != 0.0 else (final_equity + total_withdrawn - initial_equity)
    total_return_pct = (total_profit / initial_equity) * 100.0
    account_return_pct = (final_equity - initial_equity) / initial_equity * 100.0
    
    avg_profit = sum(t["net_pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t["net_pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else (float("inf") if avg_profit > 0 else 0.0)
    
    lines.append("总体统计")
    lines.append("-" * 150)
    lines.append(f"初始资金: ${initial_equity:,.2f} USDT")
    lines.append(f"账户最终资金: ${final_equity:,.2f} USDT")
    lines.append(f"累计提取金额: ${total_withdrawn:,.2f} USDT")
    lines.append(f"提取次数: {withdrawal_count}")
    lines.append(f"总收益（含提取）: ${total_profit:,.2f} USDT")
    lines.append(f"总收益率（含提取）: {total_return_pct:.2f}%")
    lines.append(f"账户收益率: {account_return_pct:.2f}%")
    
    # AI模型状态
    if ai_model_seen > 0:
        lines.append(f"AI模型训练样本数: {ai_model_seen}")
        avg_ai_score = sum(t.get("ai_score", 50.0) for t in trades) / len(trades) if trades else 50.0
        lines.append(f"平均AI评分: {avg_ai_score:.1f}")
    else:
        lines.append("AI模型状态: 未训练（所有评分固定为50.0）")
        lines.append("提示: AI模型需要在实盘交易中通过历史交易数据进行训练")
    
    lines.append(f"总交易数: {total_trades}")
    lines.append(f"盈利交易: {len(winning_trades)} ({win_rate:.1f}%)")
    lines.append(f"亏损交易: {len(losing_trades)} ({100.0 - win_rate:.1f}%)")
    lines.append(f"总盈利: ${total_pnl:,.2f} USDT")
    lines.append(f"总手续费: ${total_fee:,.2f} USDT")
    lines.append(f"净盈利: ${total_net_pnl:,.2f} USDT")
    lines.append(f"平均盈利: ${avg_profit:,.2f} USDT")
    lines.append(f"平均亏损: ${avg_loss:,.2f} USDT")
    lines.append(f"盈亏比: {profit_factor:.2f}")
    lines.append("")
    
    # 显示提取记录
    if withdrawals:
        lines.append("盈利提取记录")
        lines.append("-" * 150)
        lines.append(f"{'序号':<6} | {'提取时间':<20} | {'提取金额':>12} | {'提取前资金':>12} | {'提取后资金':>12} | {'交易序号':>8}")
        lines.append("-" * 150)
        for i, wd in enumerate(withdrawals, 1):
            lines.append(
                f"{i:<6} | "
                f"{wd['time_str']:<20} | "
                f"${wd['amount']:>11.2f} | "
                f"${wd['equity_before']:>11.2f} | "
                f"${wd['equity_after']:>11.2f} | "
                f"{wd['trade_index']:>8}"
            )
        lines.append("")
    
    # 详细交易列表
    lines.append("=" * 150)
    lines.append("详细交易记录")
    lines.append("=" * 150)
    lines.append("")
    
    # 表头
    header = (
        f"{'序号':<6} | "
        f"{'方向':<6} | "
        f"{'开仓时间':<20} | "
        f"{'平仓时间':<20} | "
        f"{'开仓价':>10} | "
        f"{'平仓价':>10} | "
        f"{'持仓天数':>10} | "
        f"{'杠杆':>6} | "
        f"{'AI评分':>8} | "
        f"{'收益率':>10} | "
        f"{'盈亏':>12} | "
        f"{'手续费':>10} | "
        f"{'盈亏比':>10} | "
        f"{'退出原因':<15}"
    )
    lines.append(header)
    lines.append("-" * 150)
    
    # 交易记录
    for i, trade in enumerate(trades, 1):
        direction_str = "做多" if trade.get("direction", "LONG") == "LONG" else "做空"
        ai_score = trade.get("ai_score", 50.0)
        row = (
            f"{i:<6} | "
            f"{direction_str:<6} | "
            f"{trade['entry_time_str']:<20} | "
            f"{trade['exit_time_str']:<20} | "
            f"${trade['entry_price_actual']:>9.2f} | "
            f"${trade['exit_price_actual']:>9.2f} | "
            f"{trade['holding_days']:>9.2f} | "
            f"{trade['leverage']:>5.1f}x | "
            f"{ai_score:>7.1f} | "
            f"{trade['return_pct']:>9.2f}% | "
            f"${trade['net_pnl']:>11.2f} | "
            f"${trade['total_fee']:>9.2f} | "
            f"{trade['actual_risk_reward']:>9.2f} | "
            f"{trade['exit_reason']:<15}"
        )
        lines.append(row)
    
    lines.append("")
    lines.append("=" * 150)
    
    # 最佳和最差交易
    if trades:
        best_trade = max(trades, key=lambda t: t["net_pnl"])
        worst_trade = min(trades, key=lambda t: t["net_pnl"])
        
        lines.append("最佳交易")
        lines.append("-" * 150)
        lines.append(f"买入时间: {best_trade['entry_time_str']}")
        lines.append(f"卖出时间: {best_trade['exit_time_str']}")
        lines.append(f"买入价: ${best_trade['entry_price_actual']:.2f}")
        lines.append(f"卖出价: ${best_trade['exit_price_actual']:.2f}")
        lines.append(f"收益率: {best_trade['return_pct']:.2f}%")
        lines.append(f"盈亏: ${best_trade['net_pnl']:.2f} USDT")
        lines.append(f"持仓天数: {best_trade['holding_days']:.2f}")
        lines.append("")
        
        lines.append("最差交易")
        lines.append("-" * 150)
        lines.append(f"买入时间: {worst_trade['entry_time_str']}")
        lines.append(f"卖出时间: {worst_trade['exit_time_str']}")
        lines.append(f"买入价: ${worst_trade['entry_price_actual']:.2f}")
        lines.append(f"卖出价: ${worst_trade['exit_price_actual']:.2f}")
        lines.append(f"收益率: {worst_trade['return_pct']:.2f}%")
        lines.append(f"盈亏: ${worst_trade['net_pnl']:.2f} USDT")
        lines.append(f"持仓天数: {worst_trade['holding_days']:.2f}")
        lines.append("")
    
    return "\n".join(lines)


def run_detailed_backtest(
    *,
    symbol: str = "BTCUSDT",
    months: int = 6,
    interval_minutes: Optional[int] = None,
    combinations: List[str],
    initial_equity_usdt: float = 1000.0,
    profit_withdrawal_threshold: float = 500.0,  # 盈利提取阈值
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.001,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> int:
    """运行详细回测"""
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
    
    trace_id = new_trace_id("backtest_detailed")
    logger.info(f"开始详细回测: symbol={symbol}, interval={interval_minutes}, trace_id={trace_id}")
    logger.info(f"时间范围: {datetime.fromtimestamp(start_time_ms/1000)} ~ {datetime.fromtimestamp(end_time_ms/1000)}")
    
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
        print(f"错误: 没有足够的数据用于回测")
        db.close()
        return 1
    
    logger.info(f"获取到 {len(rows)} 条K线数据")
    
    # 加载AI模型（如果启用）
    ai_model = None
    ai_model_seen = 0
    if runtime_cfg and runtime_cfg.ai_enabled:
        try:
            ai_model = _load_ai_model(db, settings, runtime_cfg)
            # 检查模型的训练状态
            ai_model_seen = getattr(ai_model, "seen", 0)
            if ai_model_seen == 0:
                logger.warning("AI模型已加载，但尚未训练（seen=0），所有评分将为50.0")
                logger.warning("提示：AI模型需要在实盘交易中通过 partial_fit 进行训练")
            else:
                logger.info(f"AI模型加载成功，已训练样本数: {ai_model_seen}，将使用真实AI评分")
        except Exception as e:
            logger.warning(f"AI模型加载失败，将使用默认评分50.0: {e}")
            ai_model = None
    else:
        logger.info("AI未启用（runtime_cfg.ai_enabled=False），将使用默认评分50.0")
    
    # 对每个组合进行回测
    for combination in combinations:
        condition_names = [c.strip() for c in combination.split("+")]
        logger.info(f"测试组合: {combination}")
        
        # 查找所有信号（做多和做空）
        signals = []
        for i in range(1, len(rows)):
            current = rows[i]
            prev = rows[i - 1]
            
            # 计算真实的AI评分
            ai_score = 50.0  # 默认值
            if ai_model is not None:
                try:
                    x, feat_bundle = _vectorize_for_ai(current)
                    ai_prob = float(ai_model.predict_proba(x))
                    ai_score = ai_prob * 100.0  # 转换为0-100的评分
                except Exception as e:
                    # 如果AI预测失败，使用默认值
                    ai_score = 50.0
            
            # 检查做多信号
            if check_combination(
                current,
                prev,
                condition_names=condition_names,
                runtime_cfg=runtime_cfg,
                settings=settings,
                ai_score=ai_score,
                direction="LONG",
            ):
                signals.append({"idx": i, "direction": "LONG", "ai_score": ai_score})
            
            # 检查做空信号（使用相反的条件）
            if check_combination(
                current,
                prev,
                condition_names=condition_names,
                runtime_cfg=runtime_cfg,
                settings=settings,
                ai_score=ai_score,
                direction="SHORT",
            ):
                signals.append({"idx": i, "direction": "SHORT", "ai_score": ai_score})
        
        logger.info(f"找到 {len(signals)} 个信号")
        
        if not signals:
            print(f"\n组合 {combination}: 没有找到信号")
            continue
        
        # 模拟交易（支持盈利提取策略）
        current_equity = initial_equity_usdt  # 当前可用资金
        total_withdrawn = 0.0  # 累计提取金额
        withdrawal_count = 0  # 提取次数
        withdrawals = []  # 提取记录
        trades = []
        current_position = None
        
        for signal in signals:
            if current_position is not None:
                continue
            
            signal_idx = signal["idx"]
            signal_direction = signal["direction"]
            
            kline = rows[signal_idx]
            entry_price = float(kline.get("close_price", 0))
            if entry_price <= 0:
                continue
            
            # 使用当前可用资金计算仓位
            # 计算杠杆和保证金（基于当前可用资金）
            signal_type = "BUY" if signal_direction == "LONG" else "SELL"
            score = compute_robot_score(kline, signal=signal_type)
            leverage = leverage_from_score(settings, score)
            
            # 使用信号中的AI评分（如果已计算），否则重新计算
            ai_score_for_trade = signal.get("ai_score", 50.0)
            if ai_model is not None and ai_score_for_trade == 50.0:
                try:
                    x, feat_bundle = _vectorize_for_ai(kline)
                    # 根据模型类型调用不同的predict_proba
                    proba_result = ai_model.predict_proba(x)
                    if isinstance(proba_result, list):
                        ai_prob = float(proba_result[1])  # SGDClassifierCompat返回[prob_0, prob_1]
                    else:
                        ai_prob = float(proba_result)  # OnlineLogisticRegression返回float
                    ai_score_for_trade = ai_prob * 100.0
                except Exception as e:
                    logger.debug(f"AI预测失败: {e}")
                    ai_score_for_trade = 50.0
            
            base_margin_usdt = compute_base_margin_usdt(equity_usdt=current_equity, ai_score=ai_score_for_trade, settings=settings)
            
            # 检查风险预算（基于当前可用资金）
            ok_risk, leverage_adj, _ = enforce_risk_budget(
                equity_usdt=current_equity,
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
            trade = simulate_trade_detailed(
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
                direction=signal_direction,
            )
            
            # 记录AI评分
            trade["ai_score"] = ai_score_for_trade
            
            trades.append(trade)
            
            # 更新当前可用资金
            current_equity += trade["net_pnl"]
            
            # 检查是否需要提取盈利
            # 如果盈利达到阈值（当前资金 >= 初始资金 + 阈值），提取阈值金额
            if current_equity >= initial_equity_usdt + profit_withdrawal_threshold:
                withdrawal_amount = profit_withdrawal_threshold
                current_equity -= withdrawal_amount
                total_withdrawn += withdrawal_amount
                withdrawal_count += 1
                withdrawals.append({
                    "time_ms": trade["exit_time_ms"],
                    "time_str": trade["exit_time_str"],
                    "amount": withdrawal_amount,
                    "equity_before": current_equity + withdrawal_amount,
                    "equity_after": current_equity,
                    "trade_index": len(trades),
                })
                logger.info(f"盈利提取: ${withdrawal_amount:.2f}, 提取后资金: ${current_equity:.2f}")
            
            current_position = None
        
        # 计算最终权益
        final_equity = current_equity
        total_profit = final_equity + total_withdrawn - initial_equity_usdt
        
        # 生成详细报告
        report = generate_detailed_report(
            trades,
            symbol=symbol,
            combination=combination,
            initial_equity=initial_equity_usdt,
            final_equity=final_equity,
            total_withdrawn=total_withdrawn,
            withdrawal_count=withdrawal_count,
            withdrawals=withdrawals,
            total_profit=total_profit,
            ai_model_seen=ai_model_seen,
        )
        
        print("\n" + report)
    
    db.close()
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成详细回测报告")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--months", type=int, default=6, help="回测月数")
    parser.add_argument("--interval", type=int, default=None, help="K线周期（分钟）")
    parser.add_argument("--combinations", type=str, nargs="+", required=True, help="要测试的组合，用+连接")
    parser.add_argument("--equity", type=float, default=1000.0, help="初始资金USDT")
    parser.add_argument("--fee-rate", type=float, default=0.0004, help="手续费率")
    parser.add_argument("--slippage-rate", type=float, default=0.001, help="滑点率")
    parser.add_argument("--stop-loss", type=float, default=0.02, help="止损比例")
    parser.add_argument("--take-profit", type=float, default=0.04, help="止盈比例")
    
    args = parser.parse_args()
    
    sys.exit(run_detailed_backtest(
        symbol=args.symbol,
        months=args.months,
        interval_minutes=args.interval,
        combinations=args.combinations,
        initial_equity_usdt=args.equity,
        profit_withdrawal_threshold=getattr(args, "profit_withdrawal_threshold", 500.0),
        fee_rate=getattr(args, "fee_rate"),
        slippage_rate=getattr(args, "slippage_rate"),
        stop_loss_pct=getattr(args, "stop_loss"),
        take_profit_pct=getattr(args, "take_profit"),
    ))
