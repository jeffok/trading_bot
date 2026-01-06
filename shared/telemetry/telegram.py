from __future__ import annotations

"""
Telegram 告警发送模块

目标：
- 支持 “文本 + JSON 摘要” 两段式告警
- payload 内允许包含 datetime / Decimal 等不可 JSON 序列化对象（自动转换）
- 不依赖任何外部私有方法名（避免 _send_text_message 之类不存在的问题）
- 自动按 Telegram 单条消息长度限制做分片发送

注意：
- 如果未配置 bot_token 或 chat_id，enabled() 返回 False，发送函数会静默返回
"""

import datetime
import json
from decimal import Decimal
from typing import Any, Dict, List, Optional

import requests


class Telegram:
    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: int = 10) -> None:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.timeout_seconds = int(timeout_seconds)

    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # -----------------------------
    # 内部发送：不依赖外部私有方法
    # -----------------------------

    def _send_message(self, text: str) -> None:
        """
        发送单条消息（会自动分片）。
        Telegram 单条消息长度限制约 4096，这里用更保守的 3500。
        """
        if not self.enabled():
            return

        max_len = 3500
        chunks: List[str] = []
        s = text or ""
        while len(s) > max_len:
            chunks.append(s[:max_len])
            s = s[max_len:]
        if s:
            chunks.append(s)

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        for part in chunks:
            data = {
                "chat_id": self.chat_id,
                "text": part,
                # 这里用 Markdown（非 V2），避免你再处理复杂转义
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
            try:
                resp = requests.post(url, data=data, timeout=self.timeout_seconds)
                # 即使发送失败也不要让主流程崩溃，只记录最小信息
                if resp.status_code != 200:
                    # 这里不 raise，避免影响业务
                    pass
            except Exception:
                # 网络抖动不影响业务
                pass

    # -----------------------------
    # 对外 API：send_alert / send_alert_zh
    # -----------------------------

    @staticmethod
    def _json_default(o: Any) -> Any:
        """
        json.dumps(default=...) 的兜底：
        - datetime/date -> ISO8601 字符串
        - Decimal -> float
        - 其它未知类型 -> str(o)
        """
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        if isinstance(o, Decimal):
            try:
                return float(o)
            except Exception:
                return str(o)
        return str(o)

    def send_alert(self, title: str, summary_lines: List[str], payload: Dict[str, Any], json_indent: int = 2) -> None:
        """
        发送告警：先发文本，再发 JSON 摘要。
        """
        if not self.enabled():
            return

        summary_lines = summary_lines or []
        text = "\n".join([title, *summary_lines]).strip()

        # JSON 摘要（允许 payload 中含 datetime）
        try:
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=json_indent,
                default=self._json_default,
            )
        except Exception as e:
            payload_json = json.dumps(
                {"_error": f"payload json encode failed: {str(e)}", "payload_str": str(payload)},
                ensure_ascii=False,
                sort_keys=True,
                indent=json_indent,
                default=self._json_default,
            )

        # Telegram 里以 code block 展示 JSON
        json_block = f"```json\n{payload_json}\n```"

        self._send_message(text)
        self._send_message(json_block)

    def send_alert_zh(self, title: str, summary_kv: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """
        中文化告警：summary_kv 用 key=value 展示 + JSON 摘要
        """
        if not self.enabled():
            return

        lines: List[str] = []
        for k, v in (summary_kv or {}).items():
            # 避免 None / datetime 直接进文本
            if isinstance(v, (datetime.datetime, datetime.date)):
                v2 = v.isoformat()
            else:
                v2 = v
            lines.append(f"- {k}: {v2}")

        # 统一调用 send_alert
        self.send_alert(title=title, summary_lines=lines, payload=payload, json_indent=2)
