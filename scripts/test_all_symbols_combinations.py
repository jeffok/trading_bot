#!/usr/bin/env python3
"""批量测试多个币种的不同条件组合，并生成汇总报告

使用方式:
    python3 scripts/test_all_symbols_combinations.py
    或
    docker compose exec api-service python3 /app/scripts/test_all_symbols_combinations.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 直接导入回测函数
from scripts.trading_test_tool.backtest_condition_combinations import backtest_combination
from shared.config import Settings, load_settings
from shared.db import PostgreSQL
from shared.domain.runtime_config import RuntimeConfig

# 要测试的币种列表
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BCHUSDT", "ETCUSDT", "LTCUSDT", "XRPUSDT",
    "LINKUSDT", "BNBUSDT", "DOGEUSDT", "OPUSDT", "ENSUSDT", "SOLUSDT",
    "AAVEUSDT", "ICPUSDT", "TRBUSDT", "DASHUSDT", "SUIUSDT", "AVAXUSDT"
]

# 要测试的条件组合
COMBINATIONS = [
    "adx_di",
    "volume_ratio",
    "momentum_flip",
    "squeeze_release",
    "adx_di+volume_ratio",
    "adx_di+momentum_flip",
    "adx_di+squeeze_release",
    "volume_ratio+momentum_flip",
    "squeeze_release+momentum_flip",
    "adx_di+volume_ratio+momentum_flip",
]

# 回测参数
MONTHS = 6
EQUITY = 1000.0
FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.001
STOP_LOSS = 0.02
TAKE_PROFIT = 0.04


# 全局变量：加载配置和数据库连接（避免重复加载）
_settings = None
_db = None
_runtime_cfg = None


def get_settings_and_db():
    """获取设置和数据库连接（单例模式）"""
    global _settings, _db, _runtime_cfg
    if _settings is None:
        _settings = load_settings()
        _db = PostgreSQL(_settings.postgres_url)
        _runtime_cfg = RuntimeConfig.load(_db, _settings)
    return _settings, _db, _runtime_cfg


def run_backtest(symbol: str, combination: str) -> Dict[str, Any]:
    """运行单个回测并返回结果"""
    try:
        settings, db, runtime_cfg = get_settings_and_db()
        
        # 计算时间范围
        import time
        now_ms = int(time.time() * 1000)
        start_time_ms = now_ms - (MONTHS * 30 * 24 * 60 * 60 * 1000)
        interval_minutes = int(settings.interval_minutes or 15)
        interval_ms = interval_minutes * 60 * 1000
        start_time_ms = (start_time_ms // interval_ms) * interval_ms
        end_time_ms = now_ms
        feature_version = int(settings.feature_version or 1)
        
        # 解析组合名称
        condition_names = [c.strip() for c in combination.split("+")]
        
        # 调用回测函数
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
            initial_equity_usdt=EQUITY,
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
            stop_loss_pct=STOP_LOSS,
            take_profit_pct=TAKE_PROFIT,
        )
        
        result["symbol"] = symbol
        result["combination"] = combination
        return result
        
    except Exception as e:
        return {
            "symbol": symbol,
            "combination": combination,
            "error": str(e)[:200],
            "signals": 0,
            "trades": 0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "return_pct": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }


def generate_summary_report(all_results: List[Dict[str, Any]]) -> str:
    """生成汇总报告"""
    lines = []
    lines.append("=" * 150)
    lines.append("多币种条件组合回测汇总报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试币种数: {len(SYMBOLS)}")
    lines.append(f"测试组合数: {len(COMBINATIONS)}")
    lines.append("=" * 150)
    lines.append("")
    
    # 按币种分组
    by_symbol = {}
    for result in all_results:
        symbol = result["symbol"]
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(result)
    
    # 为每个币种生成报告
    for symbol in SYMBOLS:
        if symbol not in by_symbol:
            continue
        
        results = by_symbol[symbol]
        # 按盈利排序
        results.sort(key=lambda x: x.get("net_pnl", 0.0), reverse=True)
        
        lines.append(f"\n{'=' * 150}")
        lines.append(f"币种: {symbol}")
        lines.append(f"{'=' * 150}")
        lines.append(f"{'组合':<40} | {'信号数':>8} | {'交易数':>8} | {'胜率':>8} | {'总盈利':>12} | {'收益率':>10} | {'最大回撤':>10} | {'盈亏比':>10}")
        lines.append("-" * 150)
        
        for r in results:
            combination = r.get("combination", "")
            signals = r.get("signals", 0)
            trades = r.get("trades", 0)
            win_rate = r.get("win_rate", 0.0)
            net_pnl = r.get("net_pnl", 0.0)
            return_pct = r.get("return_pct", 0.0)
            max_drawdown = r.get("max_drawdown", 0.0)
            profit_factor = r.get("profit_factor", 0.0)
            
            if "error" in r:
                lines.append(f"{combination:<40} | {'ERROR':>8} | {r.get('error', '')[:50]}")
            else:
                lines.append(
                    f"{combination:<40} | {signals:>8} | {trades:>8} | {win_rate:>7.1f}% | "
                    f"${net_pnl:>11.2f} | {return_pct:>9.2f}% | {max_drawdown:>9.2f}% | {profit_factor:>9.2f}"
                )
        
        # 推荐最佳组合
        best = results[0] if results else None
        if best and "error" not in best and best.get("trades", 0) > 0:
            lines.append("")
            lines.append(f"推荐组合: {best.get('combination', '')}")
            lines.append(f"  总盈利: ${best.get('net_pnl', 0.0):.2f} | 收益率: {best.get('return_pct', 0.0):.2f}% | "
                        f"胜率: {best.get('win_rate', 0.0):.1f}% | 最大回撤: {best.get('max_drawdown', 0.0):.2f}%")
    
    # 按组合汇总（跨币种）
    lines.append("\n" + "=" * 150)
    lines.append("按组合汇总（跨币种平均表现）")
    lines.append("=" * 150)
    lines.append(f"{'组合':<40} | {'平均盈利':>12} | {'平均收益率':>12} | {'平均胜率':>10} | {'盈利币种数':>12}")
    lines.append("-" * 150)
    
    by_combination = {}
    for result in all_results:
        if "error" in result:
            continue
        combination = result.get("combination", "")
        if combination not in by_combination:
            by_combination[combination] = []
        by_combination[combination].append(result)
    
    combination_summary = []
    for combination, results in by_combination.items():
        if not results:
            continue
        avg_pnl = sum(r.get("net_pnl", 0.0) for r in results) / len(results)
        avg_return = sum(r.get("return_pct", 0.0) for r in results) / len(results)
        avg_win_rate = sum(r.get("win_rate", 0.0) for r in results) / len(results)
        profitable_count = sum(1 for r in results if r.get("net_pnl", 0.0) > 0)
        
        combination_summary.append({
            "combination": combination,
            "avg_pnl": avg_pnl,
            "avg_return": avg_return,
            "avg_win_rate": avg_win_rate,
            "profitable_count": profitable_count,
        })
    
    combination_summary.sort(key=lambda x: x["avg_pnl"], reverse=True)
    
    for summary in combination_summary:
        lines.append(
            f"{summary['combination']:<40} | ${summary['avg_pnl']:>11.2f} | "
            f"{summary['avg_return']:>11.2f}% | {summary['avg_win_rate']:>9.1f}% | "
            f"{summary['profitable_count']:>12}/{len(SYMBOLS)}"
        )
    
    lines.append("")
    lines.append("=" * 150)
    
    return "\n".join(lines)


def main():
    """主函数"""
    print(f"开始批量测试 {len(SYMBOLS)} 个币种，每个币种测试 {len(COMBINATIONS)} 个组合")
    print(f"总共需要测试: {len(SYMBOLS) * len(COMBINATIONS)} 次")
    print("=" * 80)
    
    all_results = []
    total_tests = len(SYMBOLS) * len(COMBINATIONS)
    current_test = 0
    
    try:
        for symbol_idx, symbol in enumerate(SYMBOLS, 1):
            print(f"\n[{symbol_idx}/{len(SYMBOLS)}] 测试币种: {symbol}")
            print("-" * 80)
            
            for combo_idx, combination in enumerate(COMBINATIONS, 1):
                current_test += 1
                print(f"  [{combo_idx}/{len(COMBINATIONS)}] {combination} ... ", end="", flush=True)
                
                result = run_backtest(symbol, combination)
                all_results.append(result)
                
                if "error" in result:
                    print(f"❌ {result['error']}")
                else:
                    pnl = result.get("net_pnl", 0.0)
                    trades = result.get("trades", 0)
                    signals = result.get("signals", 0)
                    print(f"✓ 信号: {signals}, 交易: {trades}, 盈利: ${pnl:.2f}")
        
        print("\n" + "=" * 80)
        print("所有测试完成，生成汇总报告...")
        print("=" * 80)
        
        # 生成并输出报告
        report = generate_summary_report(all_results)
        print("\n" + report)
        
        # 保存到文件（可选）
        report_file = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n报告已保存到: {report_file}")
        except Exception as e:
            print(f"\n保存报告失败: {e}")
    
    finally:
        # 关闭数据库连接
        global _db
        if _db is not None:
            try:
                _db.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
