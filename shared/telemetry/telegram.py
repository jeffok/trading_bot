from __future__ import annotations

import datetime
import json
import os
from decimal import Decimal
from html import escape as html_escape
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Telegram:
    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: int = 10) -> None:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.timeout_seconds = int(timeout_seconds)

        # 是否发送 JSON 摘要（默认开启）
        self.send_json = self._get_bool_env("TELEGRAM_SEND_JSON", default=True)

    @staticmethod
    def _get_bool_env(name: str, default: bool = True) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        return default

    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # ✅ 兼容旧接口：策略引擎用 telegram.send(...)
    def send(self, text: str) -> None:
        self.send_text(text)

    def send_text(self, text: str) -> None:
        # 纯文本发送（不使用 parse_mode，避免 '_' 等触发 Markdown 解析失败）
        self._send_message(text, parse_mode=None)

    def _post_form(self, url: str, data: Dict[str, Any]) -> bool:
        try:
            body = urlencode({k: "" if v is None else str(v) for k, v in data.items()}).encode("utf-8")
            req = Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                _ = resp.read()
            return True
        except Exception:
            return False

    def _send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
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
            payload: Dict[str, Any] = {
                "chat_id": self.chat_id,
                "text": part,
                "disable_web_page_preview": "true",
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            self._post_form(url, payload)

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

    def send_alert(
        self,
        *,
        title: str,
        summary_lines: List[str],
        payload: Dict[str, Any],
        json_indent: int = 2,
    ) -> None:
        if not self.enabled():
            return

        summary_lines = summary_lines or []
        text_msg = "\n".join([str(title), *[str(x) for x in summary_lines if str(x).strip()]]).strip()

        # 1) 文本永远发送（纯文本）
        self._send_message(text_msg, parse_mode=None)

        # 2) JSON 摘要可关闭
        if not self.send_json:
            return

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

        payload_json_html = html_escape(payload_json)
        html_msg = f"<b>JSON 摘要</b>\n<pre>{payload_json_html}</pre>"
        self._send_message(html_msg, parse_mode="HTML")

    def send_alert_zh(self, *, title: str, summary_kv: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if not self.enabled():
            return

        # 需求：每条告警必须包含 HK + UTC 时间戳（便于追溯）
        kv = dict(summary_kv or {})
        if "ts_hk" not in kv:
            try:
                from zoneinfo import ZoneInfo
                kv["ts_hk"] = datetime.datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
            except Exception:
                kv["ts_hk"] = datetime.datetime.now().isoformat()
        if "ts_utc" not in kv:
            kv["ts_utc"] = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

        preferred = [
            "ts_hk","ts_utc",
            "level","service","event","action",
            "trace_id",
            "exchange","symbol","side",
            "qty","price","leverage","ai_score",
            "stop_price","stop_dist_pct",
            "reason_code","reason",
            "client_order_id","exchange_order_id",
            "stop_client_order_id","stop_exchange_order_id",
            "status","error",
        ]

        lines: List[str] = []
        used = set()
        def _fmt(v: Any) -> Any:
            if isinstance(v, (datetime.datetime, datetime.date)):
                return v.isoformat()
            if isinstance(v, Decimal):
                try:
                    return float(v)
                except Exception:
                    return str(v)
            return v

        for k in preferred:
            if k in kv:
                lines.append(f"- {k}: {_fmt(kv.get(k))}")
                used.add(k)

        for k in sorted(kv.keys()):
            if k in used:
                continue
            lines.append(f"- {k}: {_fmt(kv.get(k))}")


        self.send_alert(title=title, summary_lines=lines, payload=payload, json_indent=2)
