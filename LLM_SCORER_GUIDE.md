# LLM评分器使用指南

## 概述

LLM评分器允许您使用 ChatGPT 或 DeepSeek 来分析市场数据并给出综合评分（0-100），替代或补充传统的机器学习模型。

## 优势

1. **综合评估**：LLM可以综合考虑多个技术指标，给出更全面的评分
2. **无需训练**：不需要历史交易数据训练，配置API密钥即可使用
3. **灵活分析**：LLM可以理解复杂的市场关系，而不仅仅是数值计算
4. **支持多种信号**：可以同时评估多个信号组合的效果

## 配置方法

### 方法1：使用环境变量（推荐）

在 `.env` 文件中添加以下配置：

#### 启用/禁用LLM（重要）
```bash
LLM_ENABLED=true  # 设置为 false 可禁用LLM，直接使用传统AI模型（不尝试LLM）
LLM_FALLBACK_TO_AI=false  # 设置为 true 时，LLM失败会自动回退到传统AI模型；false时，LLM失败只使用默认评分50.0
```

#### 使用 OpenAI (ChatGPT)
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4o-mini  # 可选，默认使用 gpt-4o-mini
# 注意：gpt-5-mini 等新模型可能不支持 temperature 参数，系统会自动处理
```

#### 使用 DeepSeek
```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
LLM_MODEL=deepseek-chat  # 可选，默认使用 deepseek-chat
```

#### 可选配置
```bash
LLM_TIMEOUT=5.0  # API超时时间（秒），默认5秒
LLM_MAX_RETRIES=1  # 最大重试次数，默认1次
LLM_CACHE_ENABLED=true  # 是否启用缓存，默认true（缓存1小时）
LLM_CACHE_TTL=3600  # 缓存时间（秒），默认3600秒（1小时）
LLM_BASE_URL=  # 自定义API地址（可选）
```

#### 禁用LLM（使用传统AI模型）
如果不想使用LLM，可以设置：
```bash
LLM_ENABLED=false
```
或者直接不配置LLM相关的环境变量，系统会自动使用传统AI模型。

### 方法2：在代码中配置

```python
from shared.ai.llm_scorer import LLMScorer, LLMScorerConfig

config = LLMScorerConfig(
    provider="deepseek",  # 或 "openai"
    api_key="sk-your-api-key",
    model="deepseek-chat",
    timeout=10.0,
    cache_enabled=True,
)

llm_scorer = LLMScorer(config)
```

## 使用方法

### 在回测中使用

配置好环境变量后，直接运行回测命令即可：

```bash
docker compose exec api-service tbot backtest-detailed \
    --symbol BTCUSDT \
    --combinations "adx_di+volume_ratio" \
    --months 6
```

如果检测到LLM配置，会自动使用LLM进行评分。

### 在实盘交易中使用

LLM评分器会自动集成到 `strategy-engine` 服务中。如果配置了LLM，系统会优先使用LLM评分，否则回退到传统AI模型。

## 评分逻辑

LLM会分析以下技术指标：

- **ADX**：趋势强度（>25表示强趋势）
- **+DI / -DI**：方向指标
- **RSI**：相对强弱指标（30以下超卖，70以上超买）
- **成交量比率**：当前成交量/平均成交量（>1.5表示放量）
- **动量(MOM10)**：10期动量
- **ATR**：平均真实波幅（波动性）
- **布林带宽度**：波动性指标
- **Squeeze状态**：是否从压缩状态释放

LLM会综合考虑这些因素，给出0-100的评分：

- **90-100**：非常强的信号，多个指标高度一致
- **70-89**：较强的信号，主要指标支持
- **50-69**：中等信号，指标部分支持
- **30-49**：较弱的信号，指标支持不足
- **0-29**：很弱的信号，指标不支持或矛盾

## 性能优化

### 缓存机制

LLM评分器默认启用缓存（5分钟TTL），相同特征组合的评分会被缓存，避免重复调用API。

### 成本控制

- 使用较便宜的模型（如 `gpt-4o-mini` 或 `deepseek-chat`）
- 启用缓存减少API调用
- 调整 `LLM_TIMEOUT` 避免长时间等待

## 故障处理

如果LLM API调用失败，系统会自动回退到：
1. 传统AI模型（如果已训练）
2. 默认评分50.0

## 示例

### 检查LLM配置

```bash
# 检查环境变量
docker compose exec api-service env | grep LLM
docker compose exec api-service env | grep OPENAI
docker compose exec api-service env | grep DEEPSEEK
```

### 测试LLM评分

运行回测时，日志会显示：
```
LLM评分器已启用（ChatGPT/DeepSeek），将使用LLM进行评分
```

报告会显示：
```
AI评分器: LLM (ChatGPT/DeepSeek)
平均AI评分: 72.5 (范围: 45.0 - 95.0)
```

## 注意事项

1. **API成本**：每次评分都会调用API，注意控制成本
2. **延迟**：LLM API调用有网络延迟，可能影响实时交易速度
3. **稳定性**：确保API密钥有效，网络连接稳定
4. **缓存**：相同特征组合会使用缓存，但不同组合仍会调用API

## 与传统AI模型的对比

| 特性 | LLM评分器 | 传统AI模型 |
|------|----------|-----------|
| 需要训练 | ❌ 不需要 | ✅ 需要 |
| 综合评估 | ✅ 强 | ⚠️ 中等 |
| 成本 | ⚠️ API费用 | ✅ 免费 |
| 延迟 | ⚠️ 网络延迟 | ✅ 即时 |
| 可解释性 | ✅ 高 | ⚠️ 低 |

## 推荐配置

对于回测和策略优化，推荐使用LLM评分器：
- 可以快速评估多种信号组合
- 不需要等待模型训练
- 评分更加综合和可解释

对于实盘交易，可以：
- 使用LLM进行信号筛选
- 使用传统AI模型进行快速评分
- 两者结合使用（LLM用于重要决策，传统模型用于快速评估）
