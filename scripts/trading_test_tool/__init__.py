"""Trading Test Tool - 交易系统管理工具（仅在Docker中使用）

使用方式：
    docker compose exec execution python -m scripts.trading_test_tool <command> [args...]

命令列表：
    - prepare: 准备检查（检查配置、服务状态等）
    - status: 查看系统状态
    - diagnose: 诊断为什么没有下单
    - check: 语法检查
    - halt: 暂停交易
    - resume: 恢复交易
    - emergency-exit: 紧急退出
    - set: 设置配置
    - get: 获取配置
    - list: 列出配置
    - smoke-test: 链路自检
    - e2e-test: 端到端测试
"""
