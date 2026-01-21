#!/bin/bash
# 安装回测工具所需的依赖

set -e

echo "正在安装回测工具依赖..."

# 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 安装依赖
pip3 install -r requirements.txt

echo ""
echo "✓ 依赖安装完成！"
echo ""
echo "现在可以运行回测工具："
echo "  python3 -m scripts.trading_test_tool.backtest_with_pnl --symbol BTCUSDT --months 6"
