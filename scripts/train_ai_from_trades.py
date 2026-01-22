#!/usr/bin/env python3
"""使用历史交易数据训练AI模型的工具脚本"""
import json
import sys
import time
from typing import Dict, Any, List, Optional

from shared.db import PostgreSQL
from shared.config.loader import load_settings
from shared.domain.runtime_config import RuntimeConfig
from shared.ai.model_store import save_current_model_blob
from shared.domain.system_config import write_system_config
from shared.domain.enums import ReasonCode
from shared.logging.logger import get_logger

# 导入策略引擎中的函数
from services.strategy_engine.main import (
    _load_ai_model,
    _vectorize_for_ai,
    _parse_json_maybe,
)

# 如果导入失败，定义备用函数
def _parse_json_maybe_fallback(s: object) -> dict:
    """解析JSONB字段（可能是字符串或字典）"""
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}

logger = get_logger("train_ai")


def train_ai_from_trades(
    symbol: Optional[str] = None,
    min_trades: int = 10,
    max_trades: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """从历史交易数据训练AI模型
    
    Args:
        symbol: 只训练指定交易对的模型（None表示训练所有交易对）
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
        
        # 查询已完成的交易
        logger.info("正在查询历史交易数据...")
        
        if symbol:
            query = """
                SELECT 
                    id,
                    symbol,
                    side,
                    entry_price,
                    exit_price,
                    pnl,
                    features_json,
                    label,
                    entry_time_ms,
                    exit_time_ms,
                    created_at,
                    updated_at
                FROM trade_logs
                WHERE status = 'CLOSED'
                  AND symbol = %s
                  AND features_json IS NOT NULL
                  AND pnl IS NOT NULL
                ORDER BY exit_time_ms ASC, id ASC
            """
            params = (symbol,)
        else:
            query = """
                SELECT 
                    id,
                    symbol,
                    side,
                    entry_price,
                    exit_price,
                    pnl,
                    features_json,
                    label,
                    entry_time_ms,
                    exit_time_ms,
                    created_at,
                    updated_at
                FROM trade_logs
                WHERE status = 'CLOSED'
                  AND features_json IS NOT NULL
                  AND pnl IS NOT NULL
                ORDER BY exit_time_ms ASC, id ASC
            """
            params = ()
        
        if max_trades:
            query += f" LIMIT {max_trades}"
        
        rows = db.fetch_all(query, params)
        
        if not rows:
            logger.error("未找到已完成的交易记录")
            logger.info("提示：需要先有 status='CLOSED' 且 pnl IS NOT NULL 的交易记录")
            return 1
        
        logger.info(f"找到 {len(rows)} 笔已完成的交易")
        
        if len(rows) < min_trades:
            logger.error(f"交易数量不足：需要至少 {min_trades} 笔，实际只有 {len(rows)} 笔")
            return 1
        
        # 按交易对分组统计
        symbol_stats: Dict[str, Dict[str, int]] = {}
        for row in rows:
            sym = row['symbol']
            if sym not in symbol_stats:
                symbol_stats[sym] = {'total': 0, 'profit': 0, 'loss': 0}
            symbol_stats[sym]['total'] += 1
            pnl = float(row['pnl'] or 0.0)
            if pnl > 0:
                symbol_stats[sym]['profit'] += 1
            else:
                symbol_stats[sym]['loss'] += 1
        
        logger.info("交易统计（按交易对）：")
        for sym, stats in sorted(symbol_stats.items()):
            win_rate = (stats['profit'] / stats['total']) * 100.0 if stats['total'] > 0 else 0.0
            logger.info(f"  {sym}: 总计={stats['total']}, 盈利={stats['profit']}, 亏损={stats['loss']}, 胜率={win_rate:.1f}%")
        
        # 加载或创建AI模型
        logger.info("正在加载AI模型...")
        try:
            ai_model = _load_ai_model(db, settings, runtime_cfg)
            initial_seen = getattr(ai_model, 'seen', 0)
            logger.info(f"加载现有模型，当前训练样本数: {initial_seen}")
        except Exception as e:
            logger.warning(f"无法加载现有模型，将创建新模型: {e}")
            ai_model = _load_ai_model(db, settings, runtime_cfg)  # 这会创建新模型
            initial_seen = 0
        
        # 训练模型
        logger.info("开始训练模型...")
        trained_count = 0
        error_count = 0
        
        for i, row in enumerate(rows):
            try:
                # 解析特征
                features_json = row['features_json']
                features = _parse_json_maybe(features_json)
                
                if not features:
                    logger.debug(f"跳过交易 {row['id']}：features_json 为空")
                    continue
                
                # 从 features_json 中提取向量化的特征 x
                # 根据 _close_trade_and_train 的逻辑，features_json 中应该包含 'x' 字段
                if 'x' in features and isinstance(features['x'], list):
                    # 如果已经有向量化的特征，直接使用
                    x = [float(v) for v in features['x']]
                else:
                    # 如果没有向量化的特征，需要从 features_json 中构建
                    # 创建一个临时的行数据，包含必要的字段用于向量化
                    temp_row = {
                        'ema_fast': features.get('ema_fast', 0.0),
                        'ema_slow': features.get('ema_slow', 0.0),
                        'rsi': features.get('rsi', 50.0),
                        'features_json': features_json,
                    }
                    x, feat_bundle = _vectorize_for_ai(temp_row)
                
                # 计算标签（盈利=1，亏损=0）
                pnl = float(row['pnl'] or 0.0)
                label = 1 if pnl > 0 else 0
                
                # 训练模型
                if not dry_run:
                    ai_model.partial_fit([float(v) for v in x], int(label))
                    trained_count += 1
                else:
                    trained_count += 1
                
                # 每100笔交易打印一次进度
                if (i + 1) % 100 == 0:
                    logger.info(f"训练进度: {i + 1}/{len(rows)} ({100.0 * (i + 1) / len(rows):.1f}%)")
                    
            except Exception as e:
                error_count += 1
                logger.warning(f"训练交易 {row['id']} 时出错: {e}")
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
            trace_id = f"train_ai_{int(time.time() * 1000)}"
            
            # 保存到 ai_models 表
            try:
                save_current_model_blob(
                    db,
                    model_name=settings.ai_model_key,
                    version='trained_from_trades',
                    model_dict=ai_model.to_dict(),
                    metrics={
                        'seen': int(final_seen),
                        'trained_from': 'trade_logs',
                        'trained_count': trained_count,
                        'symbol': symbol or 'all',
                    }
                )
                logger.info("✅ 模型已保存到 ai_models 表")
            except Exception as e:
                logger.warning(f"保存到 ai_models 表失败: {e}")
            
            # 保存到 system_config 表（兼容旧版本）
            try:
                write_system_config(
                    db,
                    actor="train_ai_script",
                    key=settings.ai_model_key,
                    value=json.dumps(ai_model.to_dict(), ensure_ascii=False),
                    trace_id=trace_id,
                    reason_code=ReasonCode.AI_TRAIN.value,
                    reason=f"AI model trained from {trained_count} trades, seen={final_seen}",
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
    
    parser = argparse.ArgumentParser(description="使用历史交易数据训练AI模型")
    parser.add_argument("--symbol", type=str, default=None, help="只训练指定交易对的模型（默认：所有交易对）")
    parser.add_argument("--min-trades", type=int, default=10, help="最少需要多少笔交易才进行训练（默认：10）")
    parser.add_argument("--max-trades", type=int, default=None, help="最多使用多少笔交易（默认：使用所有）")
    parser.add_argument("--dry-run", action="store_true", help="只显示统计信息，不实际训练和保存模型")
    
    args = parser.parse_args()
    
    sys.exit(train_ai_from_trades(
        symbol=args.symbol,
        min_trades=args.min_trades,
        max_trades=args.max_trades,
        dry_run=args.dry_run,
    ))
