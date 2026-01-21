#!/bin/bash
# 批量测试多个币种的 adx_di+volume_ratio 组合

SYMBOLS=("BTCUSDT" "ETHUSDT" "BNBUSDT" "SOLUSDT" "ADAUSDT" "XRPUSDT" "DOGEUSDT" "AVAXUSDT")
MONTHS=6
COMBINATION="adx_di+volume_ratio"

echo "开始批量测试 ${#SYMBOLS[@]} 个币种的组合: $COMBINATION"
echo "=========================================="

for symbol in "${SYMBOLS[@]}"; do
    echo ""
    echo "正在测试: $symbol"
    echo "----------------------------------------"
    
    docker compose exec api-service tbot backtest-combinations \
        --symbol "$symbol" \
        --months $MONTHS \
        --combinations "$COMBINATION"
    
    echo ""
    echo "=========================================="
done

echo ""
echo "所有币种测试完成！"
