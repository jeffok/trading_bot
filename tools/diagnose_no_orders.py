#!/usr/bin/env python3
"""诊断为什么没有下单的脚本。

检查：
1. HALT_TRADING 状态
2. Setup B 条件是否满足
3. AI 选币逻辑
4. 风控限制
5. 市场数据是否正常
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config.loader import load_settings
from shared.db.mariadb import MariaDB
from shared.domain.runtime_config import load_runtime_config
from services.strategy_engine.main import (
    setup_b_decision,
    last_two_cache,
    get_flag,
    get_position,
    compute_robot_score,
    leverage_from_score,
    get_equity_usdt,
    compute_base_margin_usdt,
    enforce_risk_budget,
    min_qty_from_min_margin_usdt,
)
from shared.exchange.factory import make_exchange

def diagnose_symbol(db: MariaDB, settings, symbol: str):
    """诊断单个交易对"""
    print(f"\n{'='*60}")
    print(f"诊断交易对: {symbol}")
    print(f"{'='*60}")
    
    # 1. 检查HALT状态
    halt_trading = get_flag(db, "HALT_TRADING", "false")
    print(f"\n1. HALT状态: {halt_trading}")
    if halt_trading == "true":
        print("   ⚠️  交易已暂停！这是阻止下单的主要原因。")
        print("   解决方法: 运行 'python -m tools.admin_cli resume --by admin --reason-code ADMIN_RESUME --reason \"恢复交易\"'")
        return
    
    # 2. 检查市场数据
    latest, prev = last_two_cache(db, symbol, settings.interval_minutes, settings.feature_version)
    if not latest:
        print(f"\n2. 市场数据: ❌ 无最新数据")
        print("   解决方法: 检查data-syncer是否正常运行")
        return
    
    print(f"\n2. 市场数据: ✅ 有数据")
    print(f"   - 最新K线时间: {latest.get('open_time_ms')}")
    print(f"   - 价格: {latest.get('close_price')}")
    
    if not prev:
        print(f"   ⚠️  缺少前一根K线数据（需要用于squeeze_release和mom_flip判断）")
    
    # 3. 检查持仓
    pos = get_position(db, symbol)
    base_qty = float(pos["base_qty"]) if pos else 0.0
    print(f"\n3. 当前持仓: {base_qty}")
    if base_qty > 0:
        print("   ℹ️  已有持仓，不会开新仓")
        return
    
    # 4. 检查Setup B条件
    print(f"\n4. Setup B 条件检查:")
    
    # 获取AI评分（简化版，实际应该从AI模型获取）
    ai_score = 50.0  # 默认值
    should_buy, reason_code, reason = setup_b_decision(
        latest,
        prev,
        ai_score=ai_score,
        settings=settings,
    )
    
    print(f"   - AI评分: {ai_score} (阈值: {settings.setup_b_ai_score_min})")
    print(f"   - 决策结果: {'✅ 满足条件' if should_buy else '❌ 不满足条件'}")
    print(f"   - 原因: {reason}")
    
    if not should_buy:
        print(f"\n   ⚠️  Setup B条件不满足，这是阻止下单的主要原因之一")
        print(f"   需要同时满足以下条件:")
        print(f"   - ADX >= {settings.setup_b_adx_min}")
        print(f"   - +DI > -DI")
        print(f"   - Squeeze释放 (prev squeeze_status==1 and latest==0)")
        print(f"   - 动量由负转正 (mom10从负转正)")
        print(f"   - Volume ratio >= {settings.setup_b_vol_ratio_min}")
        print(f"   - AI score >= {settings.setup_b_ai_score_min}")
        return
    
    # 5. 检查AI选币
    print(f"\n5. AI选币检查:")
    print(f"   ⚠️  需要检查AI选币逻辑（代码中需要symbol在selected_open_symbols中）")
    
    # 6. 检查并发持仓限制
    print(f"\n6. 并发持仓限制:")
    print(f"   - 最大并发持仓: {settings.max_concurrent_positions}")
    print(f"   ℹ️  需要检查当前总持仓数")
    
    # 7. 检查风控
    print(f"\n7. 风控检查:")
    try:
        ex = make_exchange(settings, service_name="diagnose")
        equity_usdt = get_equity_usdt(ex, settings)
        print(f"   - 账户权益: {equity_usdt} USDT")
        
        score = compute_robot_score(latest, signal="BUY")
        lev = leverage_from_score(settings, score)
        print(f"   - 机器人评分: {score:.2f}")
        print(f"   - 杠杆: {lev}x")
        
        base_margin_usdt = compute_base_margin_usdt(
            equity_usdt=equity_usdt,
            ai_score=ai_score,
            settings=settings
        )
        print(f"   - 基础保证金: {base_margin_usdt} USDT")
        
        ok_risk, lev2, risk_note = enforce_risk_budget(
            equity_usdt=equity_usdt,
            base_margin_usdt=base_margin_usdt,
            leverage=int(lev),
            stop_dist_pct=float(settings.hard_stop_loss_pct),
            settings=settings,
        )
        print(f"   - 风控检查: {'✅ 通过' if ok_risk else '❌ 拒绝'}")
        if not ok_risk:
            print(f"   - 拒绝原因: {risk_note}")
        
        # 检查最小订单
        last_price = float(latest["close_price"])
        qty = min_qty_from_min_margin_usdt(
            settings.min_order_usdt, last_price, lev, precision=6
        )
        print(f"   - 最小订单量: {qty}")
        if qty <= 0:
            print(f"   ⚠️  计算出的订单量 <= 0，无法下单")
    except Exception as e:
        print(f"   ❌ 风控检查失败: {e}")

def main():
    settings = load_settings()
    db = MariaDB(
        settings.db_host,
        settings.db_port,
        settings.db_user,
        settings.db_pass,
        settings.db_name,
    )
    
    runtime_cfg = load_runtime_config(db, settings)
    
    print("="*60)
    print("交易系统诊断工具")
    print("="*60)
    print(f"\n交易所: {settings.exchange}")
    print(f"交易对数量: {len(runtime_cfg.symbols)}")
    print(f"交易对列表: {', '.join(runtime_cfg.symbols)}")
    
    # 全局检查
    halt_trading = get_flag(db, "HALT_TRADING", "false")
    emergency_exit = get_flag(db, "EMERGENCY_EXIT", "false")
    
    print(f"\n全局状态:")
    print(f"  - HALT_TRADING: {halt_trading}")
    print(f"  - EMERGENCY_EXIT: {emergency_exit}")
    
    if halt_trading == "true":
        print(f"\n⚠️  交易已暂停！这是阻止所有下单的主要原因。")
        print(f"解决方法: 运行 'python -m tools.admin_cli resume --by admin --reason-code ADMIN_RESUME --reason \"恢复交易\"'")
        return
    
    # 诊断每个交易对
    symbols = runtime_cfg.symbols or [settings.symbol]
    for symbol in symbols[:5]:  # 只诊断前5个
        try:
            diagnose_symbol(db, settings, symbol)
        except Exception as e:
            print(f"\n❌ 诊断{symbol}时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("诊断完成")
    print(f"{'='*60}")
    print("\n建议:")
    print("1. 检查日志中的 'SetupB未满足' 信息，了解具体哪些条件不满足")
    print("2. 检查AI选币逻辑，确保交易对被选中")
    print("3. 如果Setup B条件太严格，可以考虑降低阈值:")
    print(f"   - SETUP_B_ADX_MIN (当前: {settings.setup_b_adx_min})")
    print(f"   - SETUP_B_VOL_RATIO_MIN (当前: {settings.setup_b_vol_ratio_min})")
    print(f"   - SETUP_B_AI_SCORE_MIN (当前: {settings.setup_b_ai_score_min})")
    print("4. 检查市场数据是否正常更新")

if __name__ == "__main__":
    main()
