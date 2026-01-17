"""历史回测工具：分析过去半年内Setup B信号出现次数

功能：
1. 检查数据库中的历史K线数据完整性（过去半年）
2. 如果数据不足，从Bybit获取历史K线并存储
3. 计算特征指标并存储到market_data_cache
4. 分析满足Setup B条件的信号次数
5. 生成回测报告
"""

from __future__ import annotations

import json
import sys
import time
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
from services.data_syncer.main import compute_features_for_bars
from services.strategy_engine.main import setup_b_decision

logger = get_logger("backtest", "INFO")

# Bybit单次最多200条K线
BYBIT_MAX_KLINE_LIMIT = 200
# 15分钟K线，半年大约需要：180天 * 24小时 * 4条/小时 = 17280条
# 如果limit=200，需要约87次请求


def check_data_completeness(
    db: PostgreSQL,
    *,
    symbol: str,
    interval_minutes: int,
    start_time_ms: int,
    end_time_ms: int,
) -> Tuple[bool, int, int, List[int]]:
    """检查指定时间范围内的K线数据完整性
    
    Returns:
        (is_complete, expected_count, actual_count, missing_ranges)
    """
    # 计算期望的K线数量
    interval_ms = interval_minutes * 60 * 1000
    expected_count = (end_time_ms - start_time_ms) // interval_ms + 1
    
    # 查询实际存在的K线
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
    
    # 查找缺失的时间段
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
    """分批次从Bybit获取历史K线数据
    
    Returns:
        List[Kline] - 所有获取到的K线数据（已排序）
    """
    all_klines: List[Any] = []
    current_ms = start_time_ms
    interval_ms = interval_minutes * 60 * 1000
    
    logger.info(f"开始获取历史K线: symbol={symbol}, start={current_ms}, end={end_time_ms}, batch_size={batch_size}")
    
    batch_num = 0
    while current_ms <= end_time_ms:
        batch_num += 1
        try:
            # 计算本次请求的end_ms（不超过end_time_ms）
            batch_end_ms = min(current_ms + (batch_size - 1) * interval_ms, end_time_ms)
            
            logger.info(f"批次 {batch_num}: 获取 {current_ms} 到 {batch_end_ms}")
            
            # 从Bybit获取K线
            klines = exchange.fetch_klines(
                symbol=symbol,
                interval_minutes=interval_minutes,
                start_ms=current_ms,
                limit=batch_size,
            )
            
            if not klines:
                logger.warning(f"批次 {batch_num}: 未获取到K线数据，可能已到达历史数据边界")
                break
            
            # 过滤掉超出范围的数据
            valid_klines = [k for k in klines if start_time_ms <= k.open_time_ms <= end_time_ms]
            all_klines.extend(valid_klines)
            
            logger.info(f"批次 {batch_num}: 获取到 {len(klines)} 条K线，有效 {len(valid_klines)} 条")
            
            # 更新current_ms为最后一条K线的下一条
            if klines:
                last_time_ms = max(k.open_time_ms for k in klines)
                current_ms = last_time_ms + interval_ms
            else:
                break
            
            # 避免请求过快，添加小延迟
            time.sleep(0.1)
            
            # 如果获取到的数据少于batch_size，说明已经到达末尾
            if len(klines) < batch_size:
                logger.info(f"批次 {batch_num}: 数据已获取完毕（返回数量 < batch_size）")
                break
                
        except Exception as e:
            logger.error(f"批次 {batch_num}: 获取K线失败: {e}", exc_info=True)
            # 如果单次失败，尝试跳过这个时间段继续
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
    """计算并存储特征到market_data_cache表
    
    需要先从market_data读取原始K线，计算特征，然后存储到cache
    """
    # 读取market_data中缺少特征的数据
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
    
    # 准备计算特征的数据格式
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
    
    # 计算特征（使用data_syncer的逻辑）
    features_list = compute_features_for_bars(bars)
    
    # 存储到market_data_cache
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


def analyze_setup_b_signals(
    db: PostgreSQL,
    settings: Settings,
    *,
    symbol: str,
    interval_minutes: int,
    feature_version: int,
    start_time_ms: int,
    end_time_ms: int,
) -> List[Dict[str, Any]]:
    """分析满足Setup B条件的信号
    
    Returns:
        List[Dict] - 每个满足条件的信号信息
    """
    # 获取所有K线及其特征（按时间排序）
    rows = db.fetch_all(
        """
        SELECT 
            mdc.open_time_ms,
            md.close_price,
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
        
        # 使用setup_b_decision判断（AI评分设为50，回测中不依赖AI）
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
        
        should_buy, reason_code, reason = setup_b_decision(
            current_dict,
            prev_dict,
            ai_score=50.0,  # 回测中AI评分设为默认值
            settings=settings,
        )
        
        if should_buy:
            signals.append({
                "open_time_ms": int(current["open_time_ms"]),
                "close_price": float(current["close_price"]),
                "features": current_features,
                "prev_features": prev_features,
                "reason_code": reason_code.value,
                "reason": reason,
            })
    
    return signals


def generate_backtest_report(
    *,
    symbol: str,
    interval_minutes: int,
    start_time_ms: int,
    end_time_ms: int,
    total_klines: int,
    signals: List[Dict[str, Any]],
) -> str:
    """生成回测报告"""
    start_dt = datetime.fromtimestamp(start_time_ms / 1000)
    end_dt = datetime.fromtimestamp(end_time_ms / 1000)
    
    report_lines = [
        "=" * 80,
        "历史回测报告",
        "=" * 80,
        "",
        f"交易对: {symbol}",
        f"K线周期: {interval_minutes} 分钟",
        f"回测时间范围: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"总K线数量: {total_klines}",
        f"Setup B 信号数量: {len(signals)}",
        f"信号频率: {len(signals) / total_klines * 100:.2f}% ({len(signals)}/{total_klines})",
        "",
        "-" * 80,
        "信号详情:",
        "-" * 80,
    ]
    
    for i, signal in enumerate(signals[:20], 1):  # 只显示前20个信号
        dt = datetime.fromtimestamp(signal["open_time_ms"] / 1000)
        report_lines.append(
            f"{i}. 时间: {dt.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"价格: {signal['close_price']:.4f}, "
            f"原因: {signal['reason']}"
        )
    
    if len(signals) > 20:
        report_lines.append(f"... 还有 {len(signals) - 20} 个信号未显示")
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    return "\n".join(report_lines)


def run_backtest(
    *,
    symbol: str = "BTCUSDT",
    months: int = 6,
    interval_minutes: Optional[int] = None,
    feature_version: Optional[int] = None,
) -> int:
    """运行回测
    
    Returns:
        0 表示成功，非0表示失败
    """
    settings = load_settings()
    db = PostgreSQL(settings.postgres_url)
    
    # 参数配置
    interval_minutes = interval_minutes or int(settings.interval_minutes or 15)
    feature_version = feature_version or int(settings.feature_version or 1)
    
    # 计算指定月份数的时间范围
    now_ms = int(time.time() * 1000)
    months_ms = months * 30 * 24 * 60 * 60 * 1000  # 简化：N个月 ≈ N*30天
    start_time_ms = now_ms - months_ms
    
    # 对齐到K线开始时间
    interval_ms = interval_minutes * 60 * 1000
    start_time_ms = (start_time_ms // interval_ms) * interval_ms
    end_time_ms = now_ms
    
    trace_id = new_trace_id("backtest")
    logger.info(f"开始回测: symbol={symbol}, interval={interval_minutes}, trace_id={trace_id}")
    logger.info(f"时间范围: {datetime.fromtimestamp(start_time_ms/1000)} ~ {datetime.fromtimestamp(end_time_ms/1000)}")
    
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
    
    # 2. 如果数据不完整，从Bybit获取
    if not is_complete:
        logger.info("步骤2: 数据不完整，从Bybit获取历史K线...")
        exchange = make_exchange(settings, metrics=Metrics("backtest"), service_name="backtest")
        
        # 获取缺失的数据
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
    
    # 4. 分析Setup B信号
    logger.info("步骤4: 分析Setup B信号...")
    signals = analyze_setup_b_signals(
        db,
        settings,
        symbol=symbol,
        interval_minutes=interval_minutes,
        feature_version=feature_version,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    
    logger.info(f"分析完成，找到 {len(signals)} 个Setup B信号")
    
    # 5. 生成报告
    logger.info("步骤5: 生成回测报告...")
    final_count = db.fetch_one(
        """
        SELECT COUNT(*) as cnt
        FROM market_data_cache
        WHERE symbol = %s
          AND interval_minutes = %s
          AND feature_version = %s
          AND open_time_ms >= %s
          AND open_time_ms <= %s
        """,
        (symbol, interval_minutes, feature_version, start_time_ms, end_time_ms),
    )
    total_klines = int(final_count["cnt"]) if final_count else 0
    
    report = generate_backtest_report(
        symbol=symbol,
        interval_minutes=interval_minutes,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        total_klines=total_klines,
        signals=signals,
    )
    
    print("\n" + report)
    logger.info("回测完成")
    
    db.close()
    return 0