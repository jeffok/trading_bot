"""LLM-based AI scorer using ChatGPT or DeepSeek API"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import httpx
except ImportError:
    httpx = None

from shared.logging.logger import get_logger

logger = get_logger("llm_scorer")


@dataclass
class LLMScorerConfig:
    """LLM scorer configuration"""
    provider: str = "openai"  # "openai" or "deepseek"
    api_key: str = ""
    base_url: str = ""  # For DeepSeek or custom OpenAI-compatible API
    model: str = "gpt-4o-mini"  # Default model
    timeout: float = 5.0  # 减少超时时间，加快失败回退
    max_retries: int = 1  # 减少重试次数
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache for 1 hour (回测中相同特征会重复出现)


class LLMScorer:
    """LLM-based AI scorer for trading signals
    
    Uses ChatGPT or DeepSeek to analyze market features and provide a score (0-100).
    Supports persistent caching in database to avoid redundant API calls.
    """
    
    def __init__(self, config: Optional[LLMScorerConfig] = None, db: Optional[Any] = None):
        self.config = config or LLMScorerConfig()
        self._cache: Dict[str, tuple[float, float]] = {}  # key -> (score, timestamp)
        self._db = db  # Optional database connection for persistent cache
        
        # Load from environment if not provided
        if not self.config.api_key:
            if self.config.provider == "openai":
                self.config.api_key = os.getenv("OPENAI_API_KEY", "")
            elif self.config.provider == "deepseek":
                self.config.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        
        if not self.config.base_url:
            if self.config.provider == "deepseek":
                self.config.base_url = "https://api.deepseek.com/v1"
            elif self.config.provider == "openai":
                self.config.base_url = "https://api.openai.com/v1"
        
        if not self.config.api_key:
            logger.warning("LLM API key not configured, LLM scorer will return default score 50.0")
        else:
            # 验证API密钥格式（OpenAI和DeepSeek的密钥通常以sk-开头）
            if not self.config.api_key.startswith("sk-"):
                logger.warning(f"LLM API key format may be incorrect (expected 'sk-...', got '{self.config.api_key[:5]}...')")
    
    def _build_prompt(self, features: Dict[str, Any], symbol: str, direction: str = "LONG") -> str:
        """Build prompt for LLM analysis"""
        
        # Extract key features with safe None handling
        def _safe_float(key: str, default: float) -> float:
            val = features.get(key)
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        
        adx = _safe_float("adx14", 0.0)
        plus_di = _safe_float("plus_di14", 0.0)
        minus_di = _safe_float("minus_di14", 0.0)
        rsi = _safe_float("rsi", 50.0)
        volume_ratio = _safe_float("vol_ratio", 1.0)
        momentum = _safe_float("mom10", 0.0)
        atr = _safe_float("atr14", 0.0)
        bb_width = _safe_float("bb_width20", 0.0)
        
        # Get squeeze status if available
        squeeze_status = features.get("squeeze_status")
        if squeeze_status is None:
            squeeze_status = 0
        try:
            squeeze_status = int(squeeze_status)
        except (TypeError, ValueError):
            squeeze_status = 0
        squeeze_release = "是" if squeeze_status == 0 else "否"
        
        prompt = f"""你是一个专业的量化交易分析师。请分析以下市场数据，并给出一个0-100的评分，表示这个交易信号的可靠性。

交易对: {symbol}
方向: {"做多" if direction == "LONG" else "做空"}

技术指标:
- ADX: {adx:.2f} (趋势强度，>25表示强趋势)
- +DI: {plus_di:.2f} (正向趋势指标)
- -DI: {minus_di:.2f} (负向趋势指标)
- RSI: {rsi:.2f} (相对强弱指标，30以下超卖，70以上超买)
- 成交量比率: {volume_ratio:.2f} (当前成交量/平均成交量，>1.5表示放量)
- 动量(MOM10): {momentum:.2f} (10期动量，正值表示上涨动量)
- ATR: {atr:.2f} (平均真实波幅，衡量波动性)
- 布林带宽度: {bb_width:.2f} (布林带宽度，衡量波动性)
- Squeeze释放: {squeeze_release} (是否从压缩状态释放)

请综合考虑以下因素：
1. 趋势强度（ADX是否足够强）
2. 方向一致性（+DI和-DI的关系是否支持该方向）
3. 动量状态（是否有足够的动量支持）
4. 成交量确认（是否有成交量支持）
5. 市场波动性（ATR和布林带宽度是否合理）
6. Squeeze状态（是否从压缩状态释放）

请只返回一个0-100之间的数字，表示这个交易信号的可靠性评分。不要返回任何其他文字或解释。

评分标准：
- 90-100: 非常强的信号，多个指标高度一致
- 70-89: 较强的信号，主要指标支持
- 50-69: 中等信号，指标部分支持
- 30-49: 较弱的信号，指标支持不足
- 0-29: 很弱的信号，指标不支持或矛盾

请直接返回数字："""
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> Optional[float]:
        """Call LLM API and extract score"""
        if not self.config.api_key:
            return None
        
        if httpx is None:
            logger.error("httpx not installed, cannot call LLM API")
            return None
        
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        # 根据模型类型选择正确的参数
        # 某些新模型（如gpt-5-mini）使用 max_completion_tokens 而不是 max_tokens
        # 某些模型不支持 temperature 参数（只支持默认值1）
        model_lower = self.config.model.lower()
        
        # 判断是否使用 max_completion_tokens
        if "gpt-5" in model_lower or "gpt-4o" in model_lower:
            max_tokens_param = "max_completion_tokens"
        else:
            max_tokens_param = "max_tokens"
        
        # 判断是否支持 temperature 参数
        # gpt-5-mini 等新模型可能不支持 temperature，只支持默认值1
        supports_temperature = "gpt-5" not in model_lower
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的量化交易分析师。你的任务是根据技术指标分析交易信号，并给出0-100之间的评分。\n\n重要：你必须只返回一个数字（0-100之间），不要返回任何其他文字、符号或解释。例如，如果评分是75，只返回：75"
                },
                {
                    "role": "user",
                    "content": prompt + "\n\n请只返回一个0-100之间的数字，不要返回任何其他内容。"
                }
            ],
        }
        
        # 只在支持的模型上添加 temperature 参数
        if supports_temperature:
            payload["temperature"] = 0.3  # Lower temperature for more consistent results
        
        # 根据模型类型设置合适的token数量
        # 新模型可能需要更多token来返回数字
        if "gpt-5" in model_lower:
            payload[max_tokens_param] = 20  # gpt-5可能需要更多token
        else:
            payload[max_tokens_param] = 10  # 其他模型使用较少token
        
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    
                    # 检查响应状态
                    if response.status_code == 400:
                        # 400错误通常是请求格式问题，尝试获取详细错误信息
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("error", {}).get("message", "Bad Request")
                            logger.error(f"LLM API 400错误: {error_msg}")
                            logger.debug(f"请求payload: {json.dumps(payload, ensure_ascii=False)}")
                        except Exception:
                            logger.error(f"LLM API 400错误: {response.text[:200]}")
                        # 400错误不重试，直接返回None
                        return None
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract score from response
                    choices = data.get("choices", [])
                    if not choices:
                        logger.warning(f"LLM returned no choices. Full response: {json.dumps(data, ensure_ascii=False)[:500]}")
                        return None
                    
                    message = choices[0].get("message", {})
                    content = message.get("content", "").strip()
                    finish_reason = choices[0].get("finish_reason", "")
                    
                    # 检查是否因为finish_reason而截断
                    if finish_reason == "length":
                        logger.warning(f"LLM response was truncated (finish_reason=length). Content: '{content}'")
                        # 即使被截断，也尝试提取数字
                    elif finish_reason and finish_reason != "stop":
                        logger.warning(f"LLM response has unusual finish_reason: {finish_reason}. Content: '{content}'")
                    
                    # 如果内容为空，记录更多调试信息
                    if not content:
                        logger.warning(f"LLM returned empty response. Full response: {json.dumps(data, ensure_ascii=False)[:500]}")
                        return None
                    
                    # Try to extract number from response
                    import re
                    # 尝试多种模式匹配数字
                    numbers = re.findall(r'\d+\.?\d*', content)
                    if numbers:
                        score = float(numbers[0])
                        # Clamp to 0-100
                        score = max(0.0, min(100.0, score))
                        logger.debug(f"LLM returned score: {score} (from content: '{content[:50]}')")
                        return score
                    else:
                        logger.warning(f"LLM returned non-numeric response: '{content}' (length: {len(content)}, finish_reason: {finish_reason})")
                        # 尝试从完整响应中提取更多信息
                        logger.debug(f"Full LLM response: {json.dumps(data, ensure_ascii=False)[:500]}")
                        return None
                        
            except httpx.TimeoutException:
                if attempt < self.config.max_retries:
                    wait_time = (attempt + 1) * 0.5
                    logger.warning(f"LLM API timeout (attempt {attempt + 1}), retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM API timeout after {self.config.max_retries + 1} attempts")
                    return None
            except httpx.HTTPStatusError as e:
                # HTTP状态错误（如400, 401, 403等）
                if e.response.status_code in (400, 401, 403):
                    # 这些错误通常不会因为重试而解决
                    try:
                        error_data = e.response.json()
                        error_msg = error_data.get("error", {}).get("message", str(e))
                        logger.error(f"LLM API错误 ({e.response.status_code}): {error_msg}")
                    except Exception:
                        logger.error(f"LLM API错误 ({e.response.status_code}): {e}")
                    return None
                # 其他HTTP错误可以重试
                if attempt < self.config.max_retries:
                    wait_time = (attempt + 1) * 0.5
                    logger.warning(f"LLM API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM API call failed after {self.config.max_retries + 1} attempts: {e}")
                    return None
            except Exception as e:
                if attempt < self.config.max_retries:
                    wait_time = (attempt + 1) * 0.5
                    logger.warning(f"LLM API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM API call failed after {self.config.max_retries + 1} attempts: {e}")
                    return None
        
        return None
    
    def _get_cache_key(self, features: Dict[str, Any], symbol: str, direction: str, quantize: bool = True) -> str:
        """Generate cache key from features
        
        Args:
            features: Market features dictionary
            symbol: Trading pair symbol
            direction: "LONG" or "SHORT"
            quantize: If True, use coarser quantization to merge similar features
        """
        # Helper function to safely get and round a feature value
        def _safe_round(key: str, default: float, decimals: int) -> float:
            val = features.get(key)
            if val is None:
                val = default
            try:
                return round(float(val), decimals)
            except (TypeError, ValueError):
                return default
        
        if quantize:
            # 使用更粗粒度的量化，将相似的特征合并
            # 这样可以减少API调用次数
            key_features = {
                "adx14": _safe_round("adx14", 0.0, 1),  # 0.1精度
                "plus_di14": _safe_round("plus_di14", 0.0, 1),
                "minus_di14": _safe_round("minus_di14", 0.0, 1),
                "rsi": _safe_round("rsi", 50.0, 0),  # 整数精度
                "vol_ratio": _safe_round("vol_ratio", 1.0, 1),  # 0.1精度
                "mom10": _safe_round("mom10", 0.0, 1),
                "squeeze_status": features.get("squeeze_status") or 0,
            }
        else:
            # 精确匹配（用于精确缓存）
            key_features = {
                "adx14": _safe_round("adx14", 0.0, 2),
                "plus_di14": _safe_round("plus_di14", 0.0, 2),
                "minus_di14": _safe_round("minus_di14", 0.0, 2),
                "rsi": _safe_round("rsi", 50.0, 1),
                "vol_ratio": _safe_round("vol_ratio", 1.0, 2),
                "mom10": _safe_round("mom10", 0.0, 2),
                "squeeze_status": features.get("squeeze_status") or 0,
            }
        return f"{symbol}:{direction}:{json.dumps(key_features, sort_keys=True)}"
    
    def _load_from_db(self, cache_key: str) -> Optional[float]:
        """Load score from database cache"""
        if not self._db:
            return None
        
        try:
            row = self._db.fetch_one(
                "SELECT score FROM llm_score_cache WHERE cache_key = %s",
                (cache_key,)
            )
            if row and row.get("score") is not None:
                return float(row["score"])
        except Exception as e:
            logger.debug(f"Failed to load from database cache: {e}")
        
        return None
    
    def _save_to_db(self, cache_key: str, features: Dict[str, Any], symbol: str, direction: str, score: float) -> None:
        """Save score to database cache"""
        if not self._db:
            return
        
        try:
            features_json = json.dumps(features, ensure_ascii=False)
            self._db.execute(
                """
                INSERT INTO llm_score_cache (cache_key, symbol, direction, score, features_json, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                ON CONFLICT (cache_key) 
                DO UPDATE SET score = EXCLUDED.score, updated_at = CURRENT_TIMESTAMP
                """,
                (cache_key, symbol, direction, score, features_json)
            )
        except Exception as e:
            logger.debug(f"Failed to save to database cache: {e}")
    
    def score(
        self,
        features: Dict[str, Any],
        symbol: str,
        direction: str = "LONG",
    ) -> float:
        """Get AI score from LLM (0-100)
        
        Args:
            features: Market features dictionary
            symbol: Trading pair symbol
            direction: "LONG" or "SHORT"
        
        Returns:
            Score between 0-100, or 50.0 if LLM is not available
        """
        if not self.config.api_key:
            return 50.0
        
        cache_key = self._get_cache_key(features, symbol, direction, quantize=False)
        
        # 1. Check in-memory cache
        if self.config.cache_enabled:
            if cache_key in self._cache:
                cached_score, cached_time = self._cache[cache_key]
                if time.time() - cached_time < self.config.cache_ttl:
                    return cached_score
                else:
                    # Remove expired cache
                    del self._cache[cache_key]
        
        # 2. Check database cache
        if self._db:
            db_score = self._load_from_db(cache_key)
            if db_score is not None:
                # Load into in-memory cache
                if self.config.cache_enabled:
                    self._cache[cache_key] = (db_score, time.time())
                logger.debug(f"Loaded score from database cache: {db_score}")
                return db_score
        
        # 3. Call LLM API
        prompt = self._build_prompt(features, symbol, direction)
        score = self._call_llm_api(prompt)
        
        if score is None:
            logger.debug("LLM API call failed, using default score 50.0")
            return 50.0
        
        # 4. Save to caches
        if self.config.cache_enabled:
            self._cache[cache_key] = (score, time.time())
            # Clean old cache entries (keep last 100)
            if len(self._cache) > 100:
                sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_cache[:-100]:
                    del self._cache[key]
        
        # Save to database
        if self._db:
            self._save_to_db(cache_key, features, symbol, direction, score)
        
        return score
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
    
    def score_batch(
        self,
        requests: List[Tuple[Dict[str, Any], str, str]],
        max_workers: int = 10,
    ) -> Tuple[List[float], int, int]:
        """批量评分，使用并发加速和特征合并优化
        
        Args:
            requests: List of (features, symbol, direction) tuples
            max_workers: 最大并发数
        
        Returns:
            Tuple of (scores, cache_hits, api_calls):
            - scores: List of scores (0-100), 如果失败则返回50.0
            - cache_hits: 缓存命中数
            - api_calls: API调用数
        """
        if not self.config.api_key:
            return ([50.0] * len(requests), 0, 0)
        
        results = [None] * len(requests)
        cache_hits = 0
        
        # 第一阶段：检查精确缓存（使用精确量化）
        cache_misses = []
        
        for idx, (features, symbol, direction) in enumerate(requests):
            cache_key = self._get_cache_key(features, symbol, direction, quantize=False)
            found = False
            
            # 1. Check in-memory cache
            if self.config.cache_enabled:
                if cache_key in self._cache:
                    cached_score, cached_time = self._cache[cache_key]
                    if time.time() - cached_time < self.config.cache_ttl:
                        results[idx] = cached_score
                        cache_hits += 1
                        found = True
            
            # 2. Check database cache
            if not found and self._db:
                db_score = self._load_from_db(cache_key)
                if db_score is not None:
                    results[idx] = db_score
                    cache_hits += 1
                    # Load into in-memory cache
                    if self.config.cache_enabled:
                        self._cache[cache_key] = (db_score, time.time())
                    found = True
            
            if not found:
                # 需要调用API（保存原始索引）
                cache_misses.append((idx, features, symbol, direction))
        
        if not cache_misses:
            # 所有结果都来自缓存
            return (results, cache_hits, 0)
        
        # 第二阶段：特征合并优化 - 将相似的特征合并，减少API调用
        # 使用粗粒度量化键来合并相似请求
        unique_requests = {}  # {quantized_key: (features, symbol, direction, [original_indices])}
        
        for original_idx, features, symbol, direction in cache_misses:
            # 使用粗粒度量化键
            quantized_key = self._get_cache_key(features, symbol, direction, quantize=True)
            
            if quantized_key not in unique_requests:
                unique_requests[quantized_key] = (features, symbol, direction, [])
            
            # 保存原始索引（在requests列表中的位置）
            unique_requests[quantized_key][3].append(original_idx)
        
        logger.info(f"特征合并优化: {len(cache_misses)} 个请求合并为 {len(unique_requests)} 个唯一请求，减少 {len(cache_misses) - len(unique_requests)} 次API调用")
        
        # 第三阶段：检查合并后的缓存
        final_api_requests = []
        quantized_cache_hits = 0
        
        for quantized_key, (features, symbol, direction, indices) in unique_requests.items():
            found = False
            
            # 1. Check in-memory cache (quantized)
            if self.config.cache_enabled:
                if quantized_key in self._cache:
                    cached_score, cached_time = self._cache[quantized_key]
                    if time.time() - cached_time < self.config.cache_ttl:
                        # 将缓存结果应用到所有原始请求
                        for orig_idx in indices:
                            results[orig_idx] = cached_score
                        quantized_cache_hits += len(indices)
                        found = True
            
            # 2. Check database cache (use representative request's exact key)
            if not found and self._db and indices:
                # Use the representative request's features to check database
                exact_key = self._get_cache_key(features, symbol, direction, quantize=False)
                db_score = self._load_from_db(exact_key)
                if db_score is not None:
                    # 将数据库缓存结果应用到所有原始请求
                    for orig_idx in indices:
                        results[orig_idx] = db_score
                    quantized_cache_hits += len(indices)
                    # Load into in-memory cache
                    if self.config.cache_enabled:
                        self._cache[quantized_key] = (db_score, time.time())
                        self._cache[exact_key] = (db_score, time.time())
                    found = True
            
            if not found:
                # 需要调用API（只调用一次，结果会应用到所有相似请求）
                final_api_requests.append((quantized_key, features, symbol, direction, indices))
        
        if not final_api_requests:
            # 所有结果都来自缓存
            return (results, cache_hits + quantized_cache_hits, 0)
        
        logger.info(f"实际需要调用API: {len(final_api_requests)} 次（原本需要 {len(cache_misses)} 次）")
        
        # 第四阶段：并发调用LLM API（只对唯一请求调用）
        def score_single(args):
            quantized_key, features, symbol, direction, indices = args
            try:
                exact_key = self._get_cache_key(features, symbol, direction, quantize=False)
                
                prompt = self._build_prompt(features, symbol, direction)
                score = self._call_llm_api(prompt)
                if score is None:
                    return (quantized_key, 50.0)
                
                # 缓存结果（同时缓存精确和粗粒度版本）
                if self.config.cache_enabled:
                    # 缓存粗粒度版本（用于合并）
                    self._cache[quantized_key] = (score, time.time())
                    # 也缓存精确版本（用于精确匹配）
                    self._cache[exact_key] = (score, time.time())
                
                # 保存到数据库（精确版本）
                if self._db:
                    self._save_to_db(exact_key, features, symbol, direction, score)
                
                return (quantized_key, score)
            except Exception as e:
                logger.debug(f"批量评分失败: {e}")
                return (quantized_key, 50.0)
        
        # 使用线程池并发执行
        api_calls = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(score_single, args): args[0] for args in final_api_requests}
            
            for future in as_completed(futures):
                try:
                    quantized_key, score = future.result()
                    if score != 50.0:  # 非默认值表示成功调用
                        api_calls += 1
                    
                    # 将结果应用到所有使用该量化键的原始请求
                    quantized_key_from_future = futures[future]
                    for quantized_key_req, features_req, symbol_req, direction_req, indices_req in final_api_requests:
                        if quantized_key_req == quantized_key_from_future:
                            for orig_idx in indices_req:
                                results[orig_idx] = score
                            break
                except Exception as e:
                    quantized_key = futures[future]
                    logger.debug(f"批量评分任务失败 (key={quantized_key}): {e}")
                    # 失败时使用默认值
                    for quantized_key_req, _, _, _, indices_req in final_api_requests:
                        if quantized_key_req == quantized_key:
                            for orig_idx in indices_req:
                                results[orig_idx] = 50.0
                            break
        
        # 清理过期缓存
        if self.config.cache_enabled and len(self._cache) > 200:
            sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:-200]:
                del self._cache[key]
        
        total_cache_hits = cache_hits + quantized_cache_hits
        return (results, total_cache_hits, api_calls)


def create_llm_scorer_from_env(db: Optional[Any] = None) -> Optional[LLMScorer]:
    """Create LLM scorer from environment variables
    
    Args:
        db: Optional database connection for persistent caching
    """
    # 检查是否启用LLM（通过环境变量控制）
    llm_enabled = os.getenv("LLM_ENABLED", "true").lower() == "true"
    if not llm_enabled:
        logger.info("LLM评分器已禁用（LLM_ENABLED=false），将使用传统AI模型")
        return None
    
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider not in ("openai", "deepseek"):
        return None
    
    # 获取API密钥
    api_key_env = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        logger.debug(f"LLM API key not found in environment variable: {api_key_env}")
        return None
    
    # 获取模型名称
    if provider == "openai":
        default_model = "gpt-4o-mini"
    else:  # deepseek
        default_model = "deepseek-chat"
    model = os.getenv("LLM_MODEL", default_model)
    
    # 获取base_url
    base_url = os.getenv("LLM_BASE_URL", "")
    
    config = LLMScorerConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=float(os.getenv("LLM_TIMEOUT", "5.0")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        cache_enabled=os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true",
        cache_ttl=int(os.getenv("LLM_CACHE_TTL", "3600")),
    )
    
    logger.info(f"LLM评分器配置: provider={provider}, model={model}, base_url={base_url or 'default'}")
    if db:
        logger.info("LLM评分器已启用数据库持久化缓存")
    
    return LLMScorer(config, db=db)
