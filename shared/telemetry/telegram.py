from __future__ import annotations

"""
Telegram 告警发送模块（不依赖第三方库）

目标：
- 支持 “文本 + JSON 摘要” 两段式告警
- payload 内允许包含 datetime / Decimal 等不可 JSON 序列化对象（自动转换）
- 使用 Python 标准库 urllib 发送 HTTP 请求（避免 requests 依赖缺失）
- 按 Telegram 单条消息长度限制进行分片发送（保守 3500）

注意：
- 如果未配置 bot_token 或 chat_id，enabled() 返回 False，发送函数会静默返回
"""

import datetime
import json
from decimal import Decimal
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Telegram:
    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: int = 10) -> None:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.timeout_seconds = int(timeout_seconds)

    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # -----------------------------
    # 内部发送：urllib 实现
    # -----------------------------

    def _post_form(self, url: str, data: Dict[str, Any]) -> None:
        """以 application/x-www-form-urlencoded 方式 POST。失败不抛异常（告警不影响主流程）。"""
        try:
            body = urlencode({k: "" if v is None else str(v) for k, v in data.items()}).encode("utf-8")
            req = Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                _ = resp.read()  # 触发请求完成
        except Exception:
            # 告警失败不影响业务
            return

    def _send_message(self, text: str) -> None:
        """
        发送单条消息（自动分片）。
        Telegram 单条消息限制约 4096，这里用更保守的 3500。
        """
        if not self.enabled():
            return

        s = text or ""
        max_len = 3500
        parts: List[str] = []
        while len(s) > max_len:
            parts.append(s[:max_len])
            s = s[max_len:]
        if s:
            parts.append(s)

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        for part in parts:
            payload = {
                "chat_id": self.chat_id,
                "text": part,
                # 使用 Markdown（非 V2），避免复杂转义
                "parse_mode": "Markdown",
                "disable_web_page_preview": "true",
            }
            self._post_form(url, payload)

    # -----------------------------
    # JSON 序列化兜底
    # -----------------------------

    @staticmethod
    def _json_default(o: Any) -> Any:
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        if isinstance(o, Decimal):
            try:
                return float(o)
            except Exception:
                return str(o)
        return str(o)

    # -----------------------------
    # 对外 API
    # -----------------------------

    def send_alert(self, title: str, summary_lines: List[str], payload: Dict[str, Any], json_indent: int = 2) -> None:
        """发送告警：先发文本，再发 JSON 摘要。"""
        if not self.enabled():
            return

        summary_lines = summary_lines or []
        text = "\n".join([title, *summary_lines]).strip()

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

        json_block = f"```json\n{payload_json}\n```"

        self._send_message(text)
        self._send_message(json_block)

    def send_alert_zh(self, title: str, summary_kv: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """中文告警：summary_kv 用 key/value 列表 + JSON 摘要。"""
        if not self.enabled():
            return

        lines: List[str] = []
        for k, v in (summary_kv or {}).items():
            if isinstance(v, (datetime.datetime, datetime.date)):
                v2 = v.isoformat()
            else:
                v2 = v
            lines.append(f"- {k}: {v2}")

        self.send_alert(title=title, summary_lines=lines, payload=payload, json_indent=2)
