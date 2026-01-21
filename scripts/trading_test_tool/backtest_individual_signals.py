"""单独测试每个Setup B条件的回测工具

功能：
1. 可以单独测试每个Setup B条件
2. 可以测试条件的组合
3. 生成对比报告，显示每个条件的信号数量和表现

使用方式：
    python3 -m scripts.trading_test_tool.backtest_individual_signals \
        --symbol BTCUSDT \
        --months 6 \
        --test-all  # 测试所有条件
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
    setup_b_decision,
    compute_robot_score,
    leverage_from_score,
    min_qty_from_min_margin_usdt,
    enforce_risk_budget,
)

logger = get_logger("backtest_individual", "INFO")


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


def check_individual_condition(
    current_features: dict,
    prev_features: dict,
    condition_name: str,
    *,
    adx_min: float = 20.0,
    vol_min: float = 1.5,
    ai_score: float = 50.0,
    ai_min: float = 55.0,
) -> Tuple[bool, str]:
    """检查单个条件是否满足
    
    Args:
        current_features: 当前K线的特征
        prev_features: 前一根K线的特征
        condition_name: 条件名称
        adx_min: ADX最小值
        vol_min: 成交量比率最小值
        ai_score: AI评分
        ai_min: AI评分最小值
    
    Returns:
        (是否满足, 原因说明)
    """
    def _fnum(d: dict, k: str) -> Optional[float]:
        try:
            v = d.get(k)
            return float(v) if v is not None else None
        except Exception:
            return None
    
    adx = _fnum(current_features, "adx14")
    pdi = _fnum(current_features, "plus_di14")
    mdi = _fnum(current_features, "minus_di14")
    vol_ratio = _fnum(current_features, "vol_ratio")
    mom = _fnum(current_features, "mom10")
    sq = _fnum(current_features, "squeeze_status")
    mom_prev = _fnum(prev_features, "mom10")
    sq_prev = _fnum(prev_features, "squeeze_status")
    
    squeeze_release = (sq_prev == 1.0 and sq == 0.0)
    mom_flip_pos = (mom_prev is not None and mom is not None and mom_prev < 0.0 and mom > 0.0)
    
    if condition_name == "adx_di":
        # ADX >= threshold and +DI > -DI
        if adx is None or pdi is None or mdi is None:
            return False, "missing_adx_di"
        if adx < adx_min:
            return False, f"adx<{adx_min}"
        if pdi <= mdi:
            return False, "+DI<=-DI"
        return True, f"adx={adx:.1f}, +di={pdi:.1f}, -di={mdi:.1f}"
    
    elif condition_name == "squeeze_release":
        # Squeeze release
        if not squeeze_release:
            return False, f"no_squeeze_release (prev={sq_prev}, curr={sq})"
        return True, f"squeeze_release (prev={sq_prev}, curr={sq})"
    
    elif condition_name == "momentum_flip":
        # Momentum flips from negative to positive
        if not mom_flip_pos:
            return False, f"no_mom_flip (prev={mom_prev}, curr={mom})"
        return True, f"mom_flip_pos (prev={mom_prev}, curr={mom})"
    
    elif condition_name == "volume_ratio":
        # Volume ratio >= threshold
        if vol_ratio is None:
            return False, "missing_vol_ratio"
        if vol_ratio < vol_min:
            return False, f"vol_ratio<{vol_min} (actual={vol_ratio:.2f})"
        return True, f"vol_ratio={vol_ratio:.2f}"
    
    elif condition_name == "ai_score":
        # AI score >= threshold
        if float(ai_score) < ai_min:
            return False, f"ai<{ai_min} (actual={ai_score:.1f})"
        return True, f"ai_score={ai_score:.1f}"
    
    else:
        return False, f"unknown_condition: {condition_name}"


def analyze_individual_signals(
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
) -> Dict[str, List[Dict[str, Any]]]:
    """分析每个条件的信号
    
    Returns:
        Dict[condition_name, List[signals]]
    """
    # 获取所有K线及其特征
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
        return {name: [] for name in condition_names}
    
    # 获取阈值配置
    if runtime_cfg:
        adx_min = float(runtime_cfg.setup_b_adx_min)
        vol_min = float(runtime_cfg.setup_b_vol_ratio_min)
        ai_min = float(runtime_cfg.setup_b_ai_score_min)
    else:
        adx_min = float(getattr(settings, "setup_b_adx_min", 20))
        vol_min = float(getattr(settings, "setup_b_vol_ratio_min", 1.5))
        ai_min = float(getattr(settings, "setup_b_ai_score_min", 55))
    
    # 为每个条件初始化信号列表
    signals_by_condition = {name: [] for name in condition_names}
    
    total_rows = len(rows)
    logger.info(f"开始分析 {total_rows} 条K线数据，检查 {len(condition_names)} 个条件")
    
    # 逐条分析
    last_progress_log = 0
    for i in range(1, len(rows)):
        # 每处理1000条记录输出一次进度
        if i - last_progress_log >= 1000:
            progress_pct = (i / total_rows * 100.0) if total_rows > 0 else 0.0
            logger.info(f"分析进度: {i}/{total_rows} ({progress_pct:.1f}%)")
            last_progress_log = i
        current = rows[i]
        prev = rows[i - 1]
        
        current_features = _parse_json_maybe(current.get("features_json"))
        prev_features = _parse_json_maybe(prev.get("features_json"))
        
        # 计算robot_score
        robot_score = compute_robot_score(
            {
                "close_price": float(current["close_price"]),
                "ema_fast": current.get("ema_fast"),
                "ema_slow": current.get("ema_slow"),
                "rsi": current.get("rsi"),
            },
            signal="BUY"
        )
        
        # AI评分设为50（回测中不依赖AI）
        ai_score = 50.0
        
        # 检查每个条件
        for condition_name in condition_names:
            ok, reason = check_individual_condition(
                current_features,
                prev_features,
                condition_name,
                adx_min=adx_min,
                vol_min=vol_min,
                ai_score=ai_score,
                ai_min=ai_min,
            )
            
            if ok:
                signals_by_condition[condition_name].append({
                    "open_time_ms": int(current["open_time_ms"]),
                    "close_price": float(current["close_price"]),
                    "open_price": float(current["open_price"]),
                    "high_price": float(current["high_price"]),
                    "low_price": float(current["low_price"]),
                    "volume": float(current["volume"]),
                    "robot_score": robot_score,
                    "features": current_features,
                    "prev_features": prev_features,
                    "reason": reason,
                    "kline_start_index": i,
                })
    
    # 输出最终进度
    logger.info(f"分析完成，共处理 {total_rows - 1} 条K线记录")
    
    return signals_by_condition


def generate_comparison_report(
    *,
    symbol: str,
    interval_minutes: int,
    start_time_ms: int,
    end_time_ms: int,
    total_klines: int,
    signals_by_condition: Dict[str, List[Dict[str, Any]]],
) -> str:
    """生成对比报告"""
    start_dt = datetime.fromtimestamp(start_time_ms / 1000)
    end_dt = datetime.fromtimestamp(end_time_ms / 1000)
    
    report_lines = [
        "=" * 100,
        "Setup B 条件单独测试报告",
        "=" * 100,
        "",
        f"交易对: {symbol}",
        f"K线周期: {interval_minutes} 分钟",
        f"回测时间范围: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"总K线数量: {total_klines}",
        "",
        "-" * 100,
        "各条件信号统计",
        "-" * 100,
    ]
    
    # 条件说明
    condition_descriptions = {
        "adx_di": "ADX >= 阈值 且 +DI > -DI",
        "squeeze_release": "Squeeze释放 (前一根squeeze_status==1, 当前==0)",
        "momentum_flip": "动量转正 (mom10从负转正)",
        "volume_ratio": "成交量比率 >= 阈值",
        "ai_score": "AI评分 >= 阈值",
    }
    
    # 统计每个条件的信号数量
    for condition_name in sorted(signals_by_condition.keys()):
        signals = signals_by_condition[condition_name]
        count = len(signals)
        frequency = (count / total_klines * 100.0) if total_klines > 0 else 0.0
        desc = condition_descriptions.get(condition_name, condition_name)
        
        report_lines.append(
            f"{condition_name:20s} | {desc:40s} | "
            f"信号数: {count:5d} | 频率: {frequency:6.2f}% ({count}/{total_klines})"
        )
    
    report_lines.extend([
        "",
        "-" * 100,
        "条件组合分析",
        "-" * 100,
    ])
    
    # 分析条件组合
    all_condition_names = list(signals_by_condition.keys())
    if len(all_condition_names) >= 2:
        # 计算交集（同时满足多个条件的信号）
        for i, cond1 in enumerate(all_condition_names):
            for cond2 in all_condition_names[i+1:]:
                signals1 = {s["open_time_ms"] for s in signals_by_condition[cond1]}
                signals2 = {s["open_time_ms"] for s in signals_by_condition[cond2]}
                intersection = signals1 & signals2
                union = signals1 | signals2
                
                if len(union) > 0:
                    overlap_pct = (len(intersection) / len(union) * 100.0)
                    report_lines.append(
                        f"{cond1} + {cond2}: "
                        f"交集={len(intersection)}, 并集={len(union)}, "
                        f"重叠率={overlap_pct:.2f}%"
                    )
    
    # 完整Setup B（所有条件同时满足）
    if len(all_condition_names) >= 5:
        all_signals = [set(s["open_time_ms"] for s in signals_by_condition[cond]) for cond in all_condition_names]
        complete_setup_b = set.intersection(*all_signals) if all_signals else set()
        report_lines.extend([
            "",
            f"完整Setup B（所有5个条件同时满足）: {len(complete_setup_b)} 个信号",
        ])
    
    report_lines.extend([
        "",
        "-" * 100,
        "信号时间分布（前10个信号）",
        "-" * 100,
    ])
    
    # 显示每个条件的前10个信号
    for condition_name in sorted(signals_by_condition.keys()):
        signals = signals_by_condition[condition_name][:10]
        if signals:
            report_lines.append(f"\n{condition_name}:")
            for s in signals:
                dt = datetime.fromtimestamp(s["open_time_ms"] / 1000)
                report_lines.append(
                    f"  {dt.strftime('%Y-%m-%d %H:%M')} | "
                    f"价格: ${s['close_price']:.2f} | "
                    f"原因: {s['reason']}"
                )
    
    report_lines.extend([
        "",
        "=" * 100,
    ])
    
    return "\n".join(report_lines)


def run_individual_signals_test(
    *,
    symbol: str = "BTCUSDT",
    months: int = 6,
    interval_minutes: Optional[int] = None,
    feature_version: Optional[int] = None,
    condition_names: Optional[List[str]] = None,
    test_all: bool = False,
) -> int:
    """运行单独条件测试"""
    logger.info("开始初始化配置...")
    settings = load_settings()
    logger.info("配置加载完成，正在连接数据库...")
    sys.stderr.flush()
    
    db = PostgreSQL(settings.postgres_url)
    logger.info("数据库连接成功，正在加载运行时配置...")
    sys.stderr.flush()
    
    # 加载运行时配置
    runtime_cfg = RuntimeConfig.load(db, settings)
    logger.info("运行时配置加载完成")
    sys.stderr.flush()
    
    # 参数配置
    interval_minutes = interval_minutes or int(settings.interval_minutes or 15)
    feature_version = feature_version or int(settings.feature_version or 1)
    
    # 计算时间范围
    now_ms = int(time.time() * 1000)
    months_ms = months * 30 * 24 * 60 * 60 * 1000
    start_time_ms = now_ms - months_ms
    
    # 对齐到K线开始时间
    interval_ms = interval_minutes * 60 * 1000
    start_time_ms = (start_time_ms // interval_ms) * interval_ms
    end_time_ms = now_ms
    
    trace_id = new_trace_id("backtest_individual")
    logger.info(f"开始单独条件测试: symbol={symbol}, interval={interval_minutes}, trace_id={trace_id}")
    logger.info(f"时间范围: {datetime.fromtimestamp(start_time_ms/1000)} ~ {datetime.fromtimestamp(end_time_ms/1000)}")
    
    # 确定要测试的条件
    all_conditions = ["adx_di", "squeeze_release", "momentum_flip", "volume_ratio", "ai_score"]
    if test_all:
        condition_names = all_conditions
    elif condition_names is None:
        condition_names = all_conditions
    
    logger.info(f"测试条件: {', '.join(condition_names)}")
    
    # 获取所有K线数据
    logger.info("正在查询K线数据（这可能需要一些时间）...")
    
    query_start_time = time.time()
    all_kline_rows = db.fetch_all(
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
    query_duration = time.time() - query_start_time
    
    logger.info(f"K线数据查询完成，耗时 {query_duration:.2f} 秒，获取到 {len(all_kline_rows) if all_kline_rows else 0} 条记录")
    
    if not all_kline_rows:
        logger.warning("没有可用于分析的K线数据")
        db.close()
        return 1
    
    total_klines = len(all_kline_rows)
    logger.info(f"获取到 {total_klines} 条K线数据")
    
    # 分析每个条件的信号
    logger.info("分析各条件的信号...")
    analysis_start_time = time.time()
    signals_by_condition = analyze_individual_signals(
        db,
        settings,
        runtime_cfg,
        symbol=symbol,
        interval_minutes=interval_minutes,
        feature_version=feature_version,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        condition_names=condition_names,
    )
    analysis_duration = time.time() - analysis_start_time
    
    logger.info(f"信号分析完成，耗时 {analysis_duration:.2f} 秒")
    
    # 统计每个条件的信号数量
    for cond_name, signals in signals_by_condition.items():
        logger.info(f"条件 {cond_name}: 找到 {len(signals)} 个信号")
    
    # 生成报告
    logger.info("生成对比报告...")
    report = generate_comparison_report(
        symbol=symbol,
        interval_minutes=interval_minutes,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        total_klines=total_klines,
        signals_by_condition=signals_by_condition,
    )
    
    sys.stdout.write("\n" + report + "\n")
    sys.stdout.flush()
    
    logger.info("测试完成")
    # 确保日志输出被刷新
    sys.stderr.flush()
    
    db.close()
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="单独测试Setup B各条件的回测工具")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对（默认: BTCUSDT）")
    parser.add_argument("--months", type=int, default=6, help="回测月数（默认: 6）")
    parser.add_argument("--interval", type=int, default=None, help="K线周期（分钟，默认: 15）")
    parser.add_argument("--feature-version", type=int, default=None, dest="feature_version", help="特征版本（默认使用配置）")
    parser.add_argument("--conditions", type=str, nargs="+", default=None, help="要测试的条件列表（如: adx_di squeeze_release）")
    parser.add_argument("--test-all", action="store_true", help="测试所有5个条件")
    
    args = parser.parse_args()
    
    condition_names = args.conditions if args.conditions else None
    
    sys.exit(run_individual_signals_test(
        symbol=args.symbol,
        months=args.months,
        interval_minutes=args.interval,
        feature_version=args.feature_version,
        condition_names=condition_names,
        test_all=args.test_all,
    ))
