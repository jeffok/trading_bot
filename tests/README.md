
# Tests

MVP 提供最小化测试目录占位。

建议后续补充：
- migrations 的集成测试（确保可重复执行）
- 幂等下单与重复事件写入测试
- data-syncer 对重复 K 线与缺口的处理测试
- strategy-engine 在 HALT / EMERGENCY_EXIT / STOP_LOSS 情况下的行为测试
