#!/usr/bin/env python3
"""使用回测数据训练AI模型的工具脚本"""
import json
import sys
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from shared.db import PostgreSQL
from shared.config.loader import load_settings
from shared.domain.runtime_config import RuntimeConfig
from shared.ai.model_store import save_current_model_blob
from shared.domain.system_config import write_system_config
from shared.domain.enums import ReasonCode
from shared.logging.logger import get_logger

# 导入回测和策略引擎中的函数
from scripts.trading_test_tool.backtest_detailed_report import (
    check_combination,
    simulate_trade_detailed,
)
from services.strategy_engine.main import (
    _load_ai_model,
    _vectorize_for_ai,
    _parse_json_maybe,
)

logger = get_logger("train_ai_backtest")


def train_ai_from_backtest(
    symbol: str,
    combinations: List[str],
    months: int = 6,
    interval_minutes: Optional[int] = None,
    min_trades: int = 10,
    max_trades: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """从回测数据训练AI模型
    
    Args:
        symbol: 交易对符号
        combinations: 要测试的组合列表
        months: 回测月数
        interval_minutes: K线周期（分钟）
        min_trades: 最少需要多少笔交易才进行训练
        max_trades: 最多使用多少笔交易（None表示使用所有）
        dry_run: 如果为True，只显示统计信息，不实际训练和保存模型
    
    Returns:
        0表示成功，1表示失败
    """
    settings = load_settings()
    db = PostgreSQL(settings.postgres_url)
    
    try:
        # 加载运行时配置
        runtime_cfg = RuntimeConfig.load(db, settings)
        
        if not runtime_cfg.ai_enabled:
            logger.warning("AI未启用（ai_enabled=False），但将继续训练模型")
        
        # 查询K线数据
        logger.info(f"正在查询K线数据: symbol={symbol}, months={months}")
        
        if interval_minutes is None:
            interval_minutes = settings.interval_minutes
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=months * 30)
        
        rows = db.fetch_all(
            """
            SELECT 
                mdc.open_time_ms,
                md.close_price,
                md.high_price,
                md.low_price,
                md.open_price,
                md.volume,
                mdc.features_json,
                mdc.ema_fast,
                mdc.ema_slow,
                mdc.rsi
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
            (
                symbol,
                interval_minutes,
                settings.feature_version,
                int(start_time.timestamp() * 1000),
                int(end_time.timestamp() * 1000),
            ),
        )
        
        if not rows:
            logger.error(f"未找到K线数据: symbol={symbol}, interval={interval_minutes}")
            return 1
        
        logger.info(f"获取到 {len(rows)} 条K线数据")
        
        # 加载AI模型（如果启用）
        ai_model = None
        if runtime_cfg.ai_enabled:
            try:
                ai_model = _load_ai_model(db, settings, runtime_cfg)
                initial_seen = getattr(ai_model, 'seen', 0)
                logger.info(f"加载现有模型，当前训练样本数: {initial_seen}")
            except Exception as e:
                logger.warning(f"无法加载现有模型，将创建新模型: {e}")
                ai_model = _load_ai_model(db, settings, runtime_cfg)
                initial_seen = 0
        else:
            logger.info("AI未启用，将创建新模型用于训练")
            ai_model = _load_ai_model(db, settings, runtime_cfg)
            initial_seen = 0
        
        # 收集所有回测交易数据
        all_trades = []
        
        for combination in combinations:
            logger.info(f"回测组合: {combination}")
            condition_names = [c.strip() for c in combination.split("+")]
            
            # 查找所有信号
            signals = []
            for i in range(1, len(rows)):
                current = rows[i]
                prev = rows[i - 1]
                
                # 计算AI评分
                ai_score = 50.0
                if ai_model is not None:
                    try:
                        x, feat_bundle = _vectorize_for_ai(current)
                        proba_result = ai_model.predict_proba(x)
                        if isinstance(proba_result, list):
                            ai_prob = float(proba_result[1])
                        else:
                            ai_prob = float(proba_result)
                        ai_score = ai_prob * 100.0
                    except Exception:
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
                    signals.append({"idx": i, "direction": "LONG", "ai_score": ai_score, "kline": current})
                
                # 检查做空信号
                if check_combination(
                    current,
                    prev,
                    condition_names=condition_names,
                    runtime_cfg=runtime_cfg,
                    settings=settings,
                    ai_score=ai_score,
                    direction="SHORT",
                ):
                    signals.append({"idx": i, "direction": "SHORT", "ai_score": ai_score, "kline": current})
            
            logger.info(f"找到 {len(signals)} 个信号")
            
            # 模拟交易（对每个信号都进行模拟，用于训练）
            # 注意：这里不使用持仓限制，因为我们的目标是收集训练数据，而不是模拟实际交易
            for signal in signals:
                signal_idx = signal["idx"]
                signal_direction = signal["direction"]
                kline = signal["kline"]
                
                entry_price = float(kline.get("close_price", 0))
                if entry_price <= 0:
                    continue
                
                # 简化的交易模拟（用于训练）
                # 使用固定的参数
                stop_loss_pct = runtime_cfg.hard_stop_loss_pct
                take_profit_pct = stop_loss_pct * 2.0  # 止盈是止损的2倍
                fee_rate = 0.0004
                slippage_rate = 0.001
                base_margin_usdt = 1000.0  # 固定保证金
                leverage = 3  # 固定杠杆
                
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
                    direction=signal_direction,
                    klines=rows,
                    start_idx=signal_idx,
                )
                
                # 添加特征数据
                trade["features_json"] = kline.get("features_json")
                trade["kline_data"] = kline
                trade["combination"] = combination
                
                all_trades.append(trade)
        
        if not all_trades:
            logger.error("回测未生成任何交易")
            return 1
        
        logger.info(f"回测共生成 {len(all_trades)} 笔交易")
        
        # 限制交易数量
        if max_trades and len(all_trades) > max_trades:
            all_trades = all_trades[:max_trades]
            logger.info(f"限制为前 {max_trades} 笔交易")
        
        if len(all_trades) < min_trades:
            logger.error(f"交易数量不足：需要至少 {min_trades} 笔，实际只有 {len(all_trades)} 笔")
            return 1
        
        # 统计交易数据
        winning_trades = [t for t in all_trades if t.get("net_pnl", 0) > 0]
        losing_trades = [t for t in all_trades if t.get("net_pnl", 0) <= 0]
        win_rate = (len(winning_trades) / len(all_trades)) * 100.0 if all_trades else 0.0
        
        logger.info("交易统计：")
        logger.info(f"  总计: {len(all_trades)} 笔")
        logger.info(f"  盈利: {len(winning_trades)} 笔 ({win_rate:.1f}%)")
        logger.info(f"  亏损: {len(losing_trades)} 笔 ({100.0 - win_rate:.1f}%)")
        
        # 训练模型
        logger.info("开始训练模型...")
        trained_count = 0
        error_count = 0
        
        for i, trade in enumerate(all_trades):
            try:
                # 获取K线数据
                kline = trade.get("kline_data")
                if not kline:
                    continue
                
                # 向量化特征
                x, feat_bundle = _vectorize_for_ai(kline)
                
                # 计算标签（盈利=1，亏损=0）
                net_pnl = trade.get("net_pnl", 0.0)
                label = 1 if net_pnl > 0 else 0
                
                # 训练模型
                if not dry_run:
                    ai_model.partial_fit([float(v) for v in x], int(label))
                    trained_count += 1
                else:
                    trained_count += 1
                
                # 每100笔交易打印一次进度
                if (i + 1) % 100 == 0:
                    logger.info(f"训练进度: {i + 1}/{len(all_trades)} ({100.0 * (i + 1) / len(all_trades):.1f}%)")
                    
            except Exception as e:
                error_count += 1
                logger.warning(f"训练交易 {i} 时出错: {e}")
                continue
        
        final_seen = getattr(ai_model, 'seen', 0)
        logger.info(f"训练完成：成功 {trained_count} 笔，失败 {error_count} 笔")
        logger.info(f"模型训练样本数：{initial_seen} -> {final_seen} (+{final_seen - initial_seen})")
        
        if dry_run:
            logger.info("【DRY RUN模式】未实际保存模型")
            return 0
        
        # 保存模型
        logger.info("正在保存模型到数据库...")
        try:
            trace_id = f"train_ai_backtest_{int(time.time() * 1000)}"
            
            # 保存到 ai_models 表
            try:
                save_current_model_blob(
                    db,
                    model_name=settings.ai_model_key,
                    version='trained_from_backtest',
                    model_dict=ai_model.to_dict(),
                    metrics={
                        'seen': int(final_seen),
                        'trained_from': 'backtest',
                        'trained_count': trained_count,
                        'symbol': symbol,
                        'combinations': combinations,
                    }
                )
                logger.info("✅ 模型已保存到 ai_models 表")
            except Exception as e:
                logger.warning(f"保存到 ai_models 表失败: {e}")
            
            # 保存到 system_config 表
            try:
                write_system_config(
                    db,
                    actor="train_ai_backtest_script",
                    key=settings.ai_model_key,
                    value=json.dumps(ai_model.to_dict(), ensure_ascii=False),
                    trace_id=trace_id,
                    reason_code=ReasonCode.AI_TRAIN.value,
                    reason=f"AI model trained from {trained_count} backtest trades, seen={final_seen}",
                    action="AI_MODEL_UPDATE",
                )
                logger.info("✅ 模型已保存到 system_config 表")
            except Exception as e:
                logger.warning(f"保存到 system_config 表失败: {e}")
            
            logger.info("✅ 模型训练和保存完成！")
            logger.info(f"现在可以使用训练好的模型进行回测，AI评分将不再是固定的50.0")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用回测数据训练AI模型")
    parser.add_argument("--symbol", type=str, required=True, help="交易对符号（必需）")
    parser.add_argument("--combinations", type=str, nargs="+", required=True, help="要测试的组合，用+连接（必需）")
    parser.add_argument("--months", type=int, default=6, help="回测月数（默认：6）")
    parser.add_argument("--interval", type=int, default=None, dest="interval_minutes", help="K线周期（分钟，默认使用配置）")
    parser.add_argument("--min-trades", type=int, default=10, dest="min_trades", help="最少需要多少笔交易才进行训练（默认：10）")
    parser.add_argument("--max-trades", type=int, default=None, dest="max_trades", help="最多使用多少笔交易（默认：使用所有）")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="只显示统计信息，不实际训练和保存模型")
    
    args = parser.parse_args()
    
    sys.exit(train_ai_from_backtest(
        symbol=args.symbol,
        combinations=args.combinations,
        months=args.months,
        interval_minutes=args.interval_minutes,
        min_trades=args.min_trades,
        max_trades=args.max_trades,
        dry_run=args.dry_run,
    ))
