"""增强版历史回测工具：分析Setup B信号并计算盈利

功能：
1. 检查数据库中的历史K线数据完整性
2. 如果数据不足，从交易所获取历史K线并存储
3. 计算特征指标并存储到market_data_cache
4. 分析满足Setup B条件的信号
5. 模拟交易执行（开仓、止损、止盈）
6. 计算每次交易的盈利
7. 生成详细的回测报告（信号统计、盈利分析、风险指标）

作者：架构师 & 交易员
版本：v2.0
"""

from __future__ import annotations

import json
import sys
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from shared.config import Settings, load_settings
from shared.db import PostgreSQL
from shared.exchange import make_exchange
from shared.logging import get_logger, new_trace_id
from shared.telemetry import Metrics
from shared.domain.runtime_config import RuntimeConfig
from services.data_syncer.main import compute_features_for_bars
from services.strategy_engine.main import (
    setup_b_decision,
    compute_robot_score,
    leverage_from_score,
    min_qty_from_min_margin_usdt,
    compute_base_margin_usdt,
    enforce_risk_budget,
)

logger = get_logger("backtest_pnl", "INFO")

# Bybit单次最多200条K线
BYBIT_MAX_KLINE_LIMIT = 200


# ==================== 数据准备函数（复用原有逻辑）====================

def check_data_completeness(
    db: PostgreSQL,
    *,
    symbol: str,
    interval_minutes: int,
    start_time_ms: int,
    end_time_ms: int,
) -> Tuple[bool, int, int, List[int]]:
    """检查指定时间范围内的K线数据完整性"""
    interval_ms = interval_minutes * 60 * 1000
    expected_count = (end_time_ms - start_time_ms) // interval_ms + 1
    
    rows = db.fetch_all(
        """
        SELECT open_time_ms
        FROM market_data
        WHERE symbol = %s
          AND interval_minutes = %s
          AND open_time_ms >= %s
          AND open_time_ms <= %s
        ORDER BY open_time_ms ASC
        """,
        (symbol, interval_minutes, start_time_ms, end_time_ms),
    )
    
    actual_times = set(int(r["open_time_ms"]) for r in rows)
    actual_count = len(actual_times)
    
    missing_ranges: List[int] = []
    current_ms = start_time_ms
    while current_ms <= end_time_ms:
        if current_ms not in actual_times:
            missing_ranges.append(current_ms)
        current_ms += interval_ms
    
    is_complete = len(missing_ranges) == 0
    return is_complete, expected_count, actual_count, missing_ranges


def fetch_historical_klines_batch(
    exchange,
    *,
    symbol: str,
    interval_minutes: int,
    start_time_ms: int,
    end_time_ms: int,
    batch_size: int = BYBIT_MAX_KLINE_LIMIT,
) -> List[Any]:
    """分批次从交易所获取历史K线数据"""
    all_klines: List[Any] = []
    current_ms = start_time_ms
    interval_ms = interval_minutes * 60 * 1000
    
    logger.info(f"开始获取历史K线: symbol={symbol}, start={current_ms}, end={end_time_ms}, batch_size={batch_size}")
    
    batch_num = 0
    while current_ms <= end_time_ms:
        batch_num += 1
        try:
            batch_end_ms = min(current_ms + (batch_size - 1) * interval_ms, end_time_ms)
            logger.info(f"批次 {batch_num}: 获取 {current_ms} 到 {batch_end_ms}")
            
            klines = exchange.fetch_klines(
                symbol=symbol,
                interval_minutes=interval_minutes,
                start_ms=current_ms,
                limit=batch_size,
            )
            
            if not klines:
                logger.warning(f"批次 {batch_num}: 未获取到K线数据，可能已到达历史数据边界")
                break
            
            valid_klines = [k for k in klines if start_time_ms <= k.open_time_ms <= end_time_ms]
            all_klines.extend(valid_klines)
            
            logger.info(f"批次 {batch_num}: 获取到 {len(klines)} 条K线，有效 {len(valid_klines)} 条")
            
            if klines:
                last_time_ms = max(k.open_time_ms for k in klines)
                current_ms = last_time_ms + interval_ms
            else:
                break
            
            time.sleep(0.1)
            
            if len(klines) < batch_size:
                logger.info(f"批次 {batch_num}: 数据已获取完毕（返回数量 < batch_size）")
                break
                
        except Exception as e:
            logger.error(f"批次 {batch_num}: 获取K线失败: {e}", exc_info=True)
            current_ms += batch_size * interval_ms
    
    # 去重并排序
    unique_klines = {}
    for k in all_klines:
        if k.open_time_ms not in unique_klines:
            unique_klines[k.open_time_ms] = k
    all_klines = sorted(unique_klines.values(), key=lambda k: k.open_time_ms)
    
    logger.info(f"总计获取到 {len(all_klines)} 条不重复的K线数据")
    return all_klines


def store_klines_to_db(
    db: PostgreSQL,
    *,
    symbol: str,
    interval_minutes: int,
    klines: List[Any],
) -> int:
    """将K线数据存储到market_data表（幂等）"""
    if not klines:
        return 0
    
    rows = [
        (
            symbol,
            interval_minutes,
            int(k.open_time_ms),
            int(k.close_time_ms),
            float(k.open),
            float(k.high),
            float(k.low),
            float(k.close),
            float(k.volume),
        )
        for k in klines
    ]
    
    with db.tx() as cur:
        cur.executemany(
            """
            INSERT INTO market_data
              (symbol, interval_minutes, open_time_ms, close_time_ms, 
               open_price, high_price, low_price, close_price, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, interval_minutes, open_time_ms) DO UPDATE SET
              close_time_ms=EXCLUDED.close_time_ms,
              open_price=EXCLUDED.open_price,
              high_price=EXCLUDED.high_price,
              low_price=EXCLUDED.low_price,
              close_price=EXCLUDED.close_price,
              volume=EXCLUDED.volume
            """,
            rows,
        )
        inserted = cur.rowcount or 0
    
    logger.info(f"存储了 {inserted} 条K线数据到market_data表")
    return inserted


def compute_and_store_features(
    db: PostgreSQL,
    settings: Settings,
    *,
    symbol: str,
    interval_minutes: int,
    feature_version: int,
) -> int:
    """计算并存储特征到market_data_cache表"""
    rows = db.fetch_all(
        """
        SELECT 
            md.open_time_ms,
            md.close_time_ms,
            md.open_price,
            md.high_price,
            md.low_price,
            md.close_price,
            md.volume
        FROM market_data md
        LEFT JOIN market_data_cache mdc
          ON md.symbol = mdc.symbol
         AND md.interval_minutes = mdc.interval_minutes
         AND md.open_time_ms = mdc.open_time_ms
         AND mdc.feature_version = %s
        WHERE md.symbol = %s
          AND md.interval_minutes = %s
          AND mdc.open_time_ms IS NULL
        ORDER BY md.open_time_ms ASC
        LIMIT 5000
        """,
        (feature_version, symbol, interval_minutes),
    )
    
    if not rows:
        logger.info("没有需要计算特征的数据")
        return 0
    
    logger.info(f"需要计算 {len(rows)} 条K线的特征")
    
    bars = [
        {
            "open_time_ms": int(r["open_time_ms"]),
            "close_price": float(r["close_price"]),
            "high_price": float(r["high_price"]),
            "low_price": float(r["low_price"]),
            "volume": float(r["volume"]),
        }
        for r in rows
    ]
    
    features_list = compute_features_for_bars(bars)
    
    cache_rows = []
    for (ot, ema_fast, ema_slow, rsi, features_dict) in features_list:
        cache_rows.append(
            (
                symbol,
                interval_minutes,
                int(ot),
                feature_version,
                float(ema_fast) if ema_fast else None,
                float(ema_slow) if ema_slow else None,
                float(rsi) if rsi else None,
                json.dumps(features_dict, ensure_ascii=False) if features_dict else None,
            )
        )
    
    if not cache_rows:
        return 0
    
    with db.tx() as cur:
        cur.executemany(
            """
            INSERT INTO market_data_cache
              (symbol, interval_minutes, open_time_ms, feature_version,
               ema_fast, ema_slow, rsi, features_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, interval_minutes, open_time_ms, feature_version) 
            DO UPDATE SET
              ema_fast=EXCLUDED.ema_fast,
              ema_slow=EXCLUDED.ema_slow,
              rsi=EXCLUDED.rsi,
              features_json=EXCLUDED.features_json
            """,
            cache_rows,
        )
        stored = cur.rowcount or 0
    
    logger.info(f"存储了 {stored} 条特征数据到market_data_cache表")
    return stored


# ==================== 信号分析函数 ====================

def analyze_setup_b_signals(
    db: PostgreSQL,
    settings: Settings,
    runtime_cfg: RuntimeConfig,
    *,
    symbol: str,
    interval_minutes: int,
    feature_version: int,
    start_time_ms: int,
    end_time_ms: int,
) -> List[Dict[str, Any]]:
    """分析满足Setup B条件的信号
    
    Returns:
        List[Dict] - 每个满足条件的信号信息，包含完整的K线数据用于后续模拟交易
    """
    # 获取所有K线及其特征（按时间排序），包含完整的OHLCV数据
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
    
    if not rows:
        return []
    
    signals: List[Dict[str, Any]] = []
    
    # 逐条分析（需要前一根K线数据来判断squeeze_release和mom_flip）
    for i in range(1, len(rows)):
        current = rows[i]
        prev = rows[i - 1]
        
        # 解析features_json
        current_features = json.loads(current["features_json"] or "{}")
        prev_features = json.loads(prev["features_json"] or "{}")
        
        # 构建用于setup_b_decision的字典
        current_dict = {
            "features_json": current["features_json"],
            "close_price": float(current["close_price"]),
            "open_time_ms": int(current["open_time_ms"]),
        }
        prev_dict = {
            "features_json": prev["features_json"],
            "close_price": float(prev["close_price"]),
            "open_time_ms": int(prev["open_time_ms"]),
        }
        
        # 计算robot_score和leverage（用于后续交易模拟）
        robot_score = compute_robot_score(
            {
                "close_price": float(current["close_price"]),
                "ema_fast": current.get("ema_fast"),
                "ema_slow": current.get("ema_slow"),
                "rsi": current.get("rsi"),
            },
            signal="BUY"
        )
        
        # 使用setup_b_decision判断（AI评分设为50，回测中不依赖AI）
        should_buy, reason_code, reason = setup_b_decision(
            current_dict,
            prev_dict,
            ai_score=50.0,  # 回测中AI评分设为默认值
            settings=settings,
            runtime_cfg=runtime_cfg,
        )
        
        if should_buy:
            signals.append({
                "signal_index": len(signals) + 1,
                "open_time_ms": int(current["open_time_ms"]),
                "entry_price": float(current["close_price"]),  # 使用收盘价作为开仓价
                "open_price": float(current["open_price"]),
                "high_price": float(current["high_price"]),
                "low_price": float(current["low_price"]),
                "close_price": float(current["close_price"]),
                "volume": float(current["volume"]),
                "ema_fast": float(current["ema_fast"]) if current.get("ema_fast") else None,
                "ema_slow": float(current["ema_slow"]) if current.get("ema_slow") else None,
                "rsi": float(current["rsi"]) if current.get("rsi") else None,
                "robot_score": robot_score,
                "features": current_features,
                "prev_features": prev_features,
                "reason_code": reason_code.value,
                "reason": reason,
                # 保存后续K线数据的起始索引，用于模拟交易
                "kline_start_index": i,
            })
    
    return signals


# ==================== 交易模拟函数 ====================

def simulate_trade(
    *,
    signal: Dict[str, Any],
    klines_after: List[Dict[str, Any]],  # 信号后的K线数据（包含OHLCV）
    settings: Settings,
    runtime_cfg: RuntimeConfig,
    initial_equity_usdt: float,
    fee_rate: float = 0.0004,  # 默认手续费率 0.04%
    slippage_rate: float = 0.001,  # 默认滑点 0.1%
) -> Dict[str, Any]:
    """模拟单笔交易的完整生命周期
    
    Args:
        signal: 信号信息（包含entry_price, robot_score等）
        klines_after: 信号后的K线数据列表
        settings: 配置
        runtime_cfg: 运行时配置
        initial_equity_usdt: 初始资金
        fee_rate: 手续费率（默认0.04%）
        slippage_rate: 滑点率（默认0.1%）
    
    Returns:
        {
            "entry_price": float,  # 实际开仓价（含滑点）
            "entry_time_ms": int,
            "exit_price": float,  # 实际平仓价（含滑点）
            "exit_time_ms": int,
            "exit_reason": str,  # "STOP_LOSS", "TAKE_PROFIT", "SIGNAL_REVERSE", "END_OF_DATA"
            "leverage": int,
            "qty": float,
            "margin_usdt": float,  # 实际保证金
            "notional_usdt": float,  # 名义价值
            "pnl_usdt": float,  # 盈亏（USDT）
            "pnl_pct": float,  # 盈亏百分比（相对于保证金）
            "pnl_pct_notional": float,  # 盈亏百分比（相对于名义价值）
            "fee_entry_usdt": float,  # 开仓手续费
            "fee_exit_usdt": float,  # 平仓手续费
            "fee_total_usdt": float,  # 总手续费
            "net_pnl_usdt": float,  # 净盈亏（扣除手续费）
            "holding_bars": int,  # 持仓K线数量
            "holding_hours": float,  # 持仓小时数
        }
    """
    entry_price_raw = float(signal["entry_price"])
    entry_time_ms = int(signal["open_time_ms"])
    
    # 计算杠杆和仓位大小
    robot_score = float(signal.get("robot_score", 50.0))
    leverage = leverage_from_score(settings, robot_score)
    
    # 计算保证金和数量
    ai_score = 50.0  # 回测中不使用AI
    base_margin_usdt = compute_base_margin_usdt(
        equity_usdt=initial_equity_usdt,
        ai_score=ai_score,
        settings=settings,
    )
    
    # 风控检查
    ok_risk, lev_adjusted, risk_note = enforce_risk_budget(
        equity_usdt=initial_equity_usdt,
        base_margin_usdt=base_margin_usdt,
        leverage=leverage,
        stop_dist_pct=float(runtime_cfg.hard_stop_loss_pct),
        settings=settings,
        runtime_cfg=runtime_cfg,
    )
    
    if not ok_risk:
        # 风控拒绝，返回空交易
        return {
            "entry_price": None,
            "entry_time_ms": entry_time_ms,
            "exit_price": None,
            "exit_time_ms": entry_time_ms,
            "exit_reason": "RISK_REJECT",
            "leverage": leverage,
            "qty": 0.0,
            "margin_usdt": 0.0,
            "notional_usdt": 0.0,
            "pnl_usdt": 0.0,
            "pnl_pct": 0.0,
            "pnl_pct_notional": 0.0,
            "fee_entry_usdt": 0.0,
            "fee_exit_usdt": 0.0,
            "fee_total_usdt": 0.0,
            "net_pnl_usdt": 0.0,
            "holding_bars": 0,
            "holding_hours": 0.0,
            "risk_note": risk_note,
        }
    
    leverage = lev_adjusted
    
    # 计算数量
    qty = min_qty_from_min_margin_usdt(
        min_margin_usdt=runtime_cfg.min_order_usdt,
        last_price=entry_price_raw,
        leverage=leverage,
        precision=6,
    )
    
    if qty <= 0:
        return {
            "entry_price": None,
            "entry_time_ms": entry_time_ms,
            "exit_price": None,
            "exit_time_ms": entry_time_ms,
            "exit_reason": "QTY_TOO_SMALL",
            "leverage": leverage,
            "qty": 0.0,
            "margin_usdt": 0.0,
            "notional_usdt": 0.0,
            "pnl_usdt": 0.0,
            "pnl_pct": 0.0,
            "pnl_pct_notional": 0.0,
            "fee_entry_usdt": 0.0,
            "fee_exit_usdt": 0.0,
            "fee_total_usdt": 0.0,
            "net_pnl_usdt": 0.0,
            "holding_bars": 0,
            "holding_hours": 0.0,
        }
    
    # 应用滑点到开仓价（买入时向上滑点）
    entry_price = entry_price_raw * (1.0 + slippage_rate)
    
    # 计算实际保证金和名义价值（使用应用滑点后的价格）
    # 保证金 = 名义价值 / 杠杆 = (价格 * 数量) / 杠杆
    notional_usdt = qty * entry_price  # 名义价值（使用实际开仓价）
    margin_usdt = notional_usdt / leverage  # 实际保证金
    
    # 计算止损价和止盈价
    stop_dist_pct = float(runtime_cfg.hard_stop_loss_pct)
    stop_price = entry_price * (1.0 - stop_dist_pct)
    
    # 可选：设置止盈（例如2倍止损距离）
    take_profit_multiplier = 2.0  # 可以配置
    take_profit_price = entry_price * (1.0 + stop_dist_pct * take_profit_multiplier)
    
    # 开仓手续费（使用名义价值）
    fee_entry_usdt = notional_usdt * fee_rate
    
    # 模拟交易：遍历后续K线，检查止损/止盈
    interval_minutes = settings.interval_minutes
    interval_hours = interval_minutes / 60.0
    
    for i, kline in enumerate(klines_after):
        low_price = float(kline["low_price"])
        high_price = float(kline["high_price"])
        close_price = float(kline["close_price"])
        kline_time_ms = int(kline["open_time_ms"])
        
        # 检查止损（优先）
        if low_price <= stop_price:
            exit_price_raw = stop_price
            # 应用滑点到平仓价（卖出时向下滑点）
            exit_price = exit_price_raw * (1.0 - slippage_rate)
            exit_time_ms = kline_time_ms
            
            # 计算盈亏（考虑杠杆）
            # 盈亏 = (平仓价 - 开仓价) / 开仓价 * 杠杆 * 保证金
            price_change_pct = (exit_price - entry_price) / entry_price
            pnl_usdt = price_change_pct * leverage * margin_usdt
            pnl_pct = price_change_pct * leverage * 100.0
            pnl_pct_notional = price_change_pct * 100.0
            
            # 平仓手续费
            fee_exit_usdt = qty * exit_price * fee_rate
            fee_total_usdt = fee_entry_usdt + fee_exit_usdt
            net_pnl_usdt = pnl_usdt - fee_total_usdt
            
            return {
                "entry_price": entry_price,
                "entry_time_ms": entry_time_ms,
                "exit_price": exit_price,
                "exit_time_ms": exit_time_ms,
                "exit_reason": "STOP_LOSS",
                "leverage": leverage,
                "qty": qty,
                "margin_usdt": margin_usdt,
                "notional_usdt": notional_usdt,
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "pnl_usdt": pnl_usdt,
                "pnl_pct": pnl_pct,
                "pnl_pct_notional": pnl_pct_notional,
                "fee_entry_usdt": fee_entry_usdt,
                "fee_exit_usdt": fee_exit_usdt,
                "fee_total_usdt": fee_total_usdt,
                "net_pnl_usdt": net_pnl_usdt,
                "holding_bars": i + 1,
                "holding_hours": (i + 1) * interval_hours,
            }
        
        # 检查止盈
        if high_price >= take_profit_price:
            exit_price_raw = take_profit_price
            exit_price = exit_price_raw * (1.0 - slippage_rate)
            exit_time_ms = kline_time_ms
            
            price_change_pct = (exit_price - entry_price) / entry_price
            pnl_usdt = price_change_pct * leverage * margin_usdt
            pnl_pct = price_change_pct * leverage * 100.0
            pnl_pct_notional = price_change_pct * 100.0
            
            fee_exit_usdt = qty * exit_price * fee_rate
            fee_total_usdt = fee_entry_usdt + fee_exit_usdt
            net_pnl_usdt = pnl_usdt - fee_total_usdt
            
            return {
                "entry_price": entry_price,
                "entry_time_ms": entry_time_ms,
                "exit_price": exit_price,
                "exit_time_ms": exit_time_ms,
                "exit_reason": "TAKE_PROFIT",
                "leverage": leverage,
                "qty": qty,
                "margin_usdt": margin_usdt,
                "notional_usdt": notional_usdt,
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "pnl_usdt": pnl_usdt,
                "pnl_pct": pnl_pct,
                "pnl_pct_notional": pnl_pct_notional,
                "fee_entry_usdt": fee_entry_usdt,
                "fee_exit_usdt": fee_exit_usdt,
                "fee_total_usdt": fee_total_usdt,
                "net_pnl_usdt": net_pnl_usdt,
                "holding_bars": i + 1,
                "holding_hours": (i + 1) * interval_hours,
            }
        
        # 可选：检查反向信号（Setup B卖出信号）
        # 这里简化处理，如果需要可以添加反向信号检测
    
    # 数据结束，使用最后价格平仓
    if klines_after:
        last_kline = klines_after[-1]
        exit_price_raw = float(last_kline["close_price"])
        exit_price = exit_price_raw * (1.0 - slippage_rate)
        exit_time_ms = int(last_kline["open_time_ms"])
        
        pnl_notional = (exit_price - entry_price) * qty
        pnl_usdt = pnl_notional * leverage / entry_price * margin_usdt
        pnl_pct = (exit_price - entry_price) / entry_price * leverage * 100.0
        pnl_pct_notional = (exit_price - entry_price) / entry_price * 100.0
        
        fee_exit_usdt = qty * exit_price * fee_rate
        fee_total_usdt = fee_entry_usdt + fee_exit_usdt
        net_pnl_usdt = pnl_usdt - fee_total_usdt
        
        return {
            "entry_price": entry_price,
            "entry_time_ms": entry_time_ms,
            "exit_price": exit_price,
            "exit_time_ms": exit_time_ms,
            "exit_reason": "END_OF_DATA",
            "leverage": leverage,
            "qty": qty,
            "margin_usdt": margin_usdt,
            "notional_usdt": notional_usdt,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "pnl_usdt": pnl_usdt,
            "pnl_pct": pnl_pct,
            "pnl_pct_notional": pnl_pct_notional,
            "fee_entry_usdt": fee_entry_usdt,
            "fee_exit_usdt": fee_exit_usdt,
            "fee_total_usdt": fee_total_usdt,
            "net_pnl_usdt": net_pnl_usdt,
            "holding_bars": len(klines_after),
            "holding_hours": len(klines_after) * interval_hours,
        }
    else:
        # 没有后续数据
        return {
            "entry_price": entry_price,
            "entry_time_ms": entry_time_ms,
            "exit_price": entry_price,  # 假设立即平仓
            "exit_time_ms": entry_time_ms,
            "exit_reason": "NO_DATA",
            "leverage": leverage,
            "qty": qty,
            "margin_usdt": margin_usdt,
            "notional_usdt": notional_usdt,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "pnl_usdt": 0.0,
            "pnl_pct": 0.0,
            "pnl_pct_notional": 0.0,
            "fee_entry_usdt": fee_entry_usdt,
            "fee_exit_usdt": fee_entry_usdt,  # 假设立即平仓，手续费相同
            "fee_total_usdt": fee_entry_usdt * 2,
            "net_pnl_usdt": -fee_entry_usdt * 2,
            "holding_bars": 0,
            "holding_hours": 0.0,
        }


def simulate_all_trades(
    *,
    signals: List[Dict[str, Any]],
    all_klines: List[Dict[str, Any]],  # 所有K线数据（用于获取后续K线）
    settings: Settings,
    runtime_cfg: RuntimeConfig,
    initial_equity_usdt: float,
    max_concurrent_positions: int = 1,  # 回测中假设单币对，最多1个并发仓位
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.001,
) -> List[Dict[str, Any]]:
    """模拟所有信号的交易
    
    Args:
        signals: 所有信号列表
        all_klines: 所有K线数据（按时间排序）
        settings: 配置
        runtime_cfg: 运行时配置
        initial_equity_usdt: 初始资金
        max_concurrent_positions: 最大并发仓位（回测中通常设为1）
        fee_rate: 手续费率
        slippage_rate: 滑点率
    
    Returns:
        List[Dict] - 每笔交易的模拟结果
    """
    trades: List[Dict[str, Any]] = []
    current_equity = initial_equity_usdt
    
    # 创建K线时间索引，方便快速查找
    kline_time_index = {k["open_time_ms"]: i for i, k in enumerate(all_klines)}
    
    for signal in signals:
        signal_time_ms = signal["open_time_ms"]
        signal_kline_index = signal.get("kline_start_index", 0)
        
        # 获取信号后的K线数据
        klines_after = all_klines[signal_kline_index + 1:]
        
        # 如果已有持仓，跳过新信号（单币对单仓位策略）
        # 这里简化处理，实际可以根据策略调整
        
        # 模拟交易
        trade_result = simulate_trade(
            signal=signal,
            klines_after=klines_after,
            settings=settings,
            runtime_cfg=runtime_cfg,
            initial_equity_usdt=current_equity,  # 使用当前资金
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        
        # 更新资金（如果交易成功）
        if trade_result.get("entry_price") is not None:
            current_equity += trade_result.get("net_pnl_usdt", 0.0)
            trade_result["equity_after"] = current_equity
        
        trade_result["signal_index"] = signal.get("signal_index", 0)
        trade_result["signal_time_ms"] = signal_time_ms
        trades.append(trade_result)
    
    return trades


# ==================== 统计分析函数 ====================

def calculate_statistics(
    *,
    trades: List[Dict[str, Any]],
    initial_equity_usdt: float,
) -> Dict[str, Any]:
    """计算回测统计指标
    
    Returns:
        包含各种统计指标的字典
    """
    if not trades:
        return {
            "total_signals": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl_usdt": 0.0,
            "total_net_pnl_usdt": 0.0,
            "total_return_pct": 0.0,
            "avg_win_usdt": 0.0,
            "avg_loss_usdt": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_usdt": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_holding_hours": 0.0,
        }
    
    # 过滤有效交易（排除风控拒绝等）
    valid_trades = [t for t in trades if t.get("entry_price") is not None and t.get("exit_price") is not None]
    
    if not valid_trades:
        return {
            "total_signals": len(trades),
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl_usdt": 0.0,
            "total_net_pnl_usdt": 0.0,
            "total_return_pct": 0.0,
            "avg_win_usdt": 0.0,
            "avg_loss_usdt": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_usdt": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_holding_hours": 0.0,
        }
    
    # 基础统计
    total_trades = len(valid_trades)
    winning_trades = [t for t in valid_trades if t.get("net_pnl_usdt", 0.0) > 0]
    losing_trades = [t for t in valid_trades if t.get("net_pnl_usdt", 0.0) < 0]
    break_even_trades = [t for t in valid_trades if t.get("net_pnl_usdt", 0.0) == 0.0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0
    
    # 盈亏统计
    total_pnl_usdt = sum(t.get("pnl_usdt", 0.0) for t in valid_trades)
    total_net_pnl_usdt = sum(t.get("net_pnl_usdt", 0.0) for t in valid_trades)
    total_fee_usdt = sum(t.get("fee_total_usdt", 0.0) for t in valid_trades)
    total_return_pct = (total_net_pnl_usdt / initial_equity_usdt * 100.0) if initial_equity_usdt > 0 else 0.0
    
    # 平均盈亏
    avg_win_usdt = sum(t.get("net_pnl_usdt", 0.0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss_usdt = sum(t.get("net_pnl_usdt", 0.0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    # 盈亏比（Profit Factor）
    total_profit = sum(t.get("net_pnl_usdt", 0.0) for t in winning_trades)
    total_loss = abs(sum(t.get("net_pnl_usdt", 0.0) for t in losing_trades))
    profit_factor = (total_profit / total_loss) if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)
    
    # 最大回撤
    equity_curve = [initial_equity_usdt]
    for trade in valid_trades:
        equity_after = trade.get("equity_after", initial_equity_usdt)
        equity_curve.append(equity_after)
    
    max_equity = initial_equity_usdt
    max_drawdown_usdt = 0.0
    max_drawdown_pct = 0.0
    
    for equity in equity_curve:
        if equity > max_equity:
            max_equity = equity
        drawdown_usdt = max_equity - equity
        drawdown_pct = (drawdown_usdt / max_equity * 100.0) if max_equity > 0 else 0.0
        if drawdown_usdt > max_drawdown_usdt:
            max_drawdown_usdt = drawdown_usdt
            max_drawdown_pct = drawdown_pct
    
    # 夏普比率（简化版，假设无风险利率为0）
    if len(valid_trades) > 1:
        returns = [t.get("net_pnl_usdt", 0.0) / initial_equity_usdt for t in valid_trades]
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        sharpe_ratio = (avg_return / std_dev) if std_dev > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # 平均持仓时间
    avg_holding_hours = sum(t.get("holding_hours", 0.0) for t in valid_trades) / len(valid_trades) if valid_trades else 0.0
    
    # 按退出原因统计
    exit_reasons = {}
    for trade in valid_trades:
        reason = trade.get("exit_reason", "UNKNOWN")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    # 最大单笔盈亏
    max_win_usdt = max((t.get("net_pnl_usdt", 0.0) for t in valid_trades), default=0.0)
    max_loss_usdt = min((t.get("net_pnl_usdt", 0.0) for t in valid_trades), default=0.0)
    
    return {
        "total_signals": len(trades),
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "break_even_trades": len(break_even_trades),
        "win_rate": win_rate,
        "total_pnl_usdt": total_pnl_usdt,
        "total_net_pnl_usdt": total_net_pnl_usdt,
        "total_fee_usdt": total_fee_usdt,
        "total_return_pct": total_return_pct,
        "avg_win_usdt": avg_win_usdt,
        "avg_loss_usdt": avg_loss_usdt,
        "profit_factor": profit_factor,
        "max_win_usdt": max_win_usdt,
        "max_loss_usdt": max_loss_usdt,
        "max_drawdown_usdt": max_drawdown_usdt,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "avg_holding_hours": avg_holding_hours,
        "exit_reasons": exit_reasons,
        "final_equity_usdt": equity_curve[-1] if equity_curve else initial_equity_usdt,
    }


# ==================== 报告生成函数 ====================

def generate_backtest_report(
    *,
    symbol: str,
    interval_minutes: int,
    start_time_ms: int,
    end_time_ms: int,
    total_klines: int,
    signals: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    statistics: Dict[str, Any],
    initial_equity_usdt: float,
) -> str:
    """生成详细的回测报告"""
    start_dt = datetime.fromtimestamp(start_time_ms / 1000)
    end_dt = datetime.fromtimestamp(end_time_ms / 1000)
    
    report_lines = [
        "=" * 100,
        "历史回测报告 - Setup B 策略",
        "=" * 100,
        "",
        f"交易对: {symbol}",
        f"K线周期: {interval_minutes} 分钟",
        f"回测时间范围: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"回测天数: {(end_time_ms - start_time_ms) / 1000 / 86400:.1f} 天",
        f"总K线数量: {total_klines}",
        f"初始资金: ${initial_equity_usdt:,.2f} USDT",
        "",
        "-" * 100,
        "信号统计",
        "-" * 100,
        f"Setup B 信号数量: {statistics['total_signals']}",
        f"信号频率: {statistics['total_signals'] / total_klines * 100:.2f}% ({statistics['total_signals']}/{total_klines})",
        "",
        "-" * 100,
        "交易统计",
        "-" * 100,
        f"总交易次数: {statistics['total_trades']}",
        f"盈利交易: {statistics['winning_trades']} ({statistics['win_rate']:.2f}%)",
        f"亏损交易: {statistics['losing_trades']} ({100 - statistics['win_rate']:.2f}%)",
        f"平均持仓时间: {statistics['avg_holding_hours']:.2f} 小时",
        "",
        "-" * 100,
        "盈亏分析",
        "-" * 100,
        f"总盈亏（未扣手续费）: ${statistics['total_pnl_usdt']:,.2f} USDT",
        f"总手续费: ${statistics['total_fee_usdt']:,.2f} USDT",
        f"净盈亏（扣除手续费）: ${statistics['total_net_pnl_usdt']:,.2f} USDT",
        f"总收益率: {statistics['total_return_pct']:.2f}%",
        f"最终资金: ${statistics['final_equity_usdt']:,.2f} USDT",
        "",
        f"平均盈利: ${statistics['avg_win_usdt']:,.2f} USDT",
        f"平均亏损: ${statistics['avg_loss_usdt']:,.2f} USDT",
        f"盈亏比（Profit Factor）: {statistics['profit_factor']:.2f}",
        f"最大单笔盈利: ${statistics['max_win_usdt']:,.2f} USDT",
        f"最大单笔亏损: ${statistics['max_loss_usdt']:,.2f} USDT",
        "",
        "-" * 100,
        "风险指标",
        "-" * 100,
        f"最大回撤: ${statistics['max_drawdown_usdt']:,.2f} USDT ({statistics['max_drawdown_pct']:.2f}%)",
        f"夏普比率: {statistics['sharpe_ratio']:.4f}",
        "",
        "-" * 100,
        "退出原因统计",
        "-" * 100,
    ]
    
    for reason, count in statistics.get("exit_reasons", {}).items():
        pct = (count / statistics['total_trades'] * 100.0) if statistics['total_trades'] > 0 else 0.0
        report_lines.append(f"  {reason}: {count} 次 ({pct:.1f}%)")
    
    report_lines.extend([
        "",
        "-" * 100,
        "交易明细（前20笔）",
        "-" * 100,
    ])
    
    # 显示前20笔交易
    valid_trades = [t for t in trades if t.get("entry_price") is not None]
    for i, trade in enumerate(valid_trades[:20], 1):
        entry_dt = datetime.fromtimestamp(trade["entry_time_ms"] / 1000)
        exit_dt = datetime.fromtimestamp(trade["exit_time_ms"] / 1000)
        net_pnl = trade.get("net_pnl_usdt", 0.0)
        pnl_pct = trade.get("pnl_pct", 0.0)
        exit_reason = trade.get("exit_reason", "UNKNOWN")
        
        report_lines.append(
            f"{i}. {entry_dt.strftime('%Y-%m-%d %H:%M')} -> {exit_dt.strftime('%Y-%m-%d %H:%M')} | "
            f"开仓: ${trade['entry_price']:.4f} | 平仓: ${trade['exit_price']:.4f} | "
            f"杠杆: {trade['leverage']}x | 持仓: {trade['holding_hours']:.1f}h | "
            f"盈亏: ${net_pnl:,.2f} ({pnl_pct:+.2f}%) | 原因: {exit_reason}"
        )
    
    if len(valid_trades) > 20:
        report_lines.append(f"... 还有 {len(valid_trades) - 20} 笔交易未显示")
    
    report_lines.extend([
        "",
        "=" * 100,
    ])
    
    return "\n".join(report_lines)


# ==================== 主函数 ====================

def run_backtest_with_pnl(
    *,
    symbol: str = "BTCUSDT",
    months: int = 6,
    interval_minutes: Optional[int] = None,
    feature_version: Optional[int] = None,
    initial_equity_usdt: float = 1000.0,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.001,
    max_lookahead_bars: int = 100,  # 最多向前看多少根K线（避免未来数据泄露）
) -> int:
    """运行带盈利计算的回测
    
    Args:
        symbol: 交易对
        months: 回测月数
        interval_minutes: K线周期（分钟）
        feature_version: 特征版本
        initial_equity_usdt: 初始资金（USDT）
        fee_rate: 手续费率（默认0.04%）
        slippage_rate: 滑点率（默认0.1%）
        max_lookahead_bars: 最多向前看的K线数量
    
    Returns:
        0 表示成功，非0表示失败
    """
    settings = load_settings()
    db = PostgreSQL(settings.postgres_url)
    runtime_cfg = RuntimeConfig.load(db, settings)
    
    # 参数配置
    interval_minutes = interval_minutes or int(settings.interval_minutes or 15)
    feature_version = feature_version or int(settings.feature_version or 1)
    
    # 计算指定月份数的时间范围
    now_ms = int(time.time() * 1000)
    months_ms = months * 30 * 24 * 60 * 60 * 1000
    start_time_ms = now_ms - months_ms
    
    # 对齐到K线开始时间
    interval_ms = interval_minutes * 60 * 1000
    start_time_ms = (start_time_ms // interval_ms) * interval_ms
    end_time_ms = now_ms
    
    trace_id = new_trace_id("backtest_pnl")
    logger.info(f"开始回测: symbol={symbol}, interval={interval_minutes}, trace_id={trace_id}")
    logger.info(f"时间范围: {datetime.fromtimestamp(start_time_ms/1000)} ~ {datetime.fromtimestamp(end_time_ms/1000)}")
    logger.info(f"初始资金: ${initial_equity_usdt:,.2f} USDT")
    
    # 1. 检查数据完整性
    logger.info("步骤1: 检查数据完整性...")
    is_complete, expected_count, actual_count, missing_ranges = check_data_completeness(
        db,
        symbol=symbol,
        interval_minutes=interval_minutes,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    
    logger.info(f"数据完整性: 期望 {expected_count} 条, 实际 {actual_count} 条, 缺失 {len(missing_ranges)} 条")
    
    # 2. 如果数据不完整，从交易所获取
    if not is_complete:
        logger.info("步骤2: 数据不完整，从交易所获取历史K线...")
        exchange = make_exchange(settings, metrics=Metrics("backtest"), service_name="backtest")
        
        klines = fetch_historical_klines_batch(
            exchange,
            symbol=symbol,
            interval_minutes=interval_minutes,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
        
        if klines:
            store_klines_to_db(db, symbol=symbol, interval_minutes=interval_minutes, klines=klines)
        else:
            logger.warning("未能获取到K线数据")
    else:
        logger.info("步骤2: 数据已完整，跳过获取")
    
    # 3. 计算并存储特征
    logger.info("步骤3: 计算并存储特征...")
    stored_features = 0
    while True:
        n = compute_and_store_features(
            db,
            settings,
            symbol=symbol,
            interval_minutes=interval_minutes,
            feature_version=feature_version,
        )
        stored_features += n
        if n == 0:
            break
    
    logger.info(f"特征计算完成，共存储 {stored_features} 条特征")
    
    # 4. 获取所有K线数据（用于交易模拟）
    logger.info("步骤4: 获取所有K线数据...")
    all_kline_rows = db.fetch_all(
        """
        SELECT 
            md.open_time_ms,
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
    
    all_klines = [
        {
            "open_time_ms": int(r["open_time_ms"]),
            "close_price": float(r["close_price"]),
            "open_price": float(r["open_price"]),
            "high_price": float(r["high_price"]),
            "low_price": float(r["low_price"]),
            "volume": float(r["volume"]),
            "ema_fast": float(r["ema_fast"]) if r.get("ema_fast") else None,
            "ema_slow": float(r["ema_slow"]) if r.get("ema_slow") else None,
            "rsi": float(r["rsi"]) if r.get("rsi") else None,
            "features_json": r["features_json"],
        }
        for r in all_kline_rows
    ]
    
    logger.info(f"获取到 {len(all_klines)} 条K线数据")
    
    # 5. 分析Setup B信号
    logger.info("步骤5: 分析Setup B信号...")
    signals = analyze_setup_b_signals(
        db,
        settings,
        runtime_cfg,
        symbol=symbol,
        interval_minutes=interval_minutes,
        feature_version=feature_version,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    
    logger.info(f"分析完成，找到 {len(signals)} 个Setup B信号")
    
    # 6. 模拟交易
    logger.info("步骤6: 模拟交易执行...")
    trades = simulate_all_trades(
        signals=signals,
        all_klines=all_klines,
        settings=settings,
        runtime_cfg=runtime_cfg,
        initial_equity_usdt=initial_equity_usdt,
        max_concurrent_positions=1,  # 单币对单仓位
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    
    logger.info(f"交易模拟完成，共 {len(trades)} 笔交易")
    
    # 7. 计算统计指标
    logger.info("步骤7: 计算统计指标...")
    statistics = calculate_statistics(
        trades=trades,
        initial_equity_usdt=initial_equity_usdt,
    )
    
    # 8. 生成报告
    logger.info("步骤8: 生成回测报告...")
    total_klines = len(all_klines)
    
    report = generate_backtest_report(
        symbol=symbol,
        interval_minutes=interval_minutes,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        total_klines=total_klines,
        signals=signals,
        trades=trades,
        statistics=statistics,
        initial_equity_usdt=initial_equity_usdt,
    )
    
    print("\n" + report)
    
    # 9. 保存详细结果到JSON（可选）
    results_file = Path(__file__).parent / f"backtest_results_{symbol}_{int(time.time())}.json"
    results_data = {
        "symbol": symbol,
        "interval_minutes": interval_minutes,
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "initial_equity_usdt": initial_equity_usdt,
        "statistics": statistics,
        "signals": signals,
        "trades": trades,
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"详细结果已保存到: {results_file}")
    logger.info("回测完成")
    
    db.close()
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="历史回测工具（带盈利计算）")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对（默认: BTCUSDT）")
    parser.add_argument("--months", type=int, default=6, help="回测月数（默认: 6）")
    parser.add_argument("--interval", type=int, default=None, help="K线周期（分钟，默认: 15）")
    parser.add_argument("--equity", type=float, default=1000.0, help="初始资金USDT（默认: 1000）")
    parser.add_argument("--fee-rate", type=float, default=0.0004, help="手续费率（默认: 0.0004 = 0.04%）")
    parser.add_argument("--slippage-rate", type=float, default=0.001, help="滑点率（默认: 0.001 = 0.1%）")
    
    args = parser.parse_args()
    
    sys.exit(run_backtest_with_pnl(
        symbol=args.symbol,
        months=args.months,
        interval_minutes=args.interval,
        initial_equity_usdt=args.equity,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    ))
