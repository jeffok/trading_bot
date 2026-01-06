from __future__ import annotations

"""
Telegram 告警工具

你的需求：
- 只把“发到 Telegram 里看到的文本”变成中文（标题/摘要/JSON展示）
- 系统内部仍可使用英文 key（level/event/service/...），不影响 DB、逻辑、统计等
"""

import html
import json
from typing import Any, Dict, Iterable, Tuple

import httpx


class Telegram:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()

    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # -------------------------
    # 低层发送
    # -------------------------
    def send_text(self, text: str) -> None:
        """发送纯文本（无 parse_mode）"""
        if not self.enabled():
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            with httpx.Client(timeout=10) as client:
                client.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "disable_web_page_preview": True,
                    },
                )
        except Exception:
            # 告警失败不能影响交易主循环
            pass

    def send_html(self, html_text: str) -> None:
        """使用 Telegram HTML parse_mode 发送（用于代码块/加粗）"""
        if not self.enabled():
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            with httpx.Client(timeout=10) as client:
                client.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": html_text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
        except Exception:
            pass

    # -------------------------
    # 标准化告警（原始）
    # -------------------------
    def send_alert(
        self,
        *,
        title: str,
        summary_lines: Iterable[str],
        payload: Dict[str, Any],
        json_indent: int = 2,
        max_len: int = 3900,
    ) -> None:
        """发送：标题 + 摘要 + JSON（内容按你传入的 payload 原样展示）"""
        if not self.enabled():
            return

        summary = "\n".join([str(x) for x in summary_lines if str(x).strip()])

        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=json_indent)

        # HTML escape 防止格式错乱/注入
        title_e = html.escape(str(title))
        summary_e = html.escape(summary)
        payload_e = html.escape(payload_json)

        head = f"<b>{title_e}</b>\n{summary_e}\n\n<pre><code>"
        tail = "</code></pre>"
        msg = head + payload_e + tail

        # Telegram 长度限制保护：只截断 JSON 部分
        if len(msg) > max_len:
            allowed = max(0, max_len - len(head) - len(tail) - 40)
            payload_trunc = payload_e[:allowed] + "\n...（内容过长已截断）"
            msg = head + payload_trunc + tail

        self.send_html(msg)

    # -------------------------
    # 你的需求：Telegram 展示中文化
    # -------------------------
    _KEY_ZH_MAP: Dict[str, str] = {
        "level": "级别",
        "event": "事件",
        "service": "服务",
        "trace_id": "追踪ID",
        "exchange": "交易所",
        "symbol": "交易对",
        "reason_code": "原因码",
        "reason": "原因说明",
        "side": "方向",
        "qty": "数量",
        "price": "价格",
        "entry_price": "开仓价",
        "last_price": "最新价",
        "stop_price": "止损价",
        "avg_entry_price": "均价",
        "client_order_id": "客户端订单ID",
        "exchange_order_id": "交易所订单ID",
        "error": "错误详情",
        "note": "说明",
        "ts": "时间",
        "ts_ms": "时间戳毫秒",
    }

    _LEVEL_ZH_MAP: Dict[str, str] = {
        "INFO": "信息",
        "WARN": "警告",
        "WARNING": "警告",
        "ERROR": "错误",
        "CRITICAL": "严重",
    }

    _EVENT_ZH_MAP: Dict[str, str] = {
        "BUY_FILLED": "开仓成交",
        "SELL_FILLED": "平仓成交",
        "STOP_LOSS": "触发止损",
        "EMERGENCY_EXIT_EXECUTED": "紧急退出已执行",
        "HALT_SKIP_TICK": "暂停中-跳过本轮",
        "NO_MARKET_DATA": "无行情数据-跳过本轮",
        "UNHANDLED_EXCEPTION": "未处理异常",
        "DATA_SYNC_ERROR": "行情同步错误",
        "ADMIN_HALT": "管理-暂停交易",
        "ADMIN_RESUME": "管理-恢复交易",
        "ADMIN_EMERGENCY_EXIT": "管理-紧急退出",
        "ADMIN_UPDATE_CONFIG": "管理-修改配置",
        "MIGRATIONS": "数据库迁移",
    }

    _SIDE_ZH_MAP: Dict[str, str] = {
        "BUY": "买入",
        "SELL": "卖出",
    }

    _REASON_CODE_ZH_MAP: Dict[str, str] = {
        "STRATEGY_SIGNAL": "策略信号",
        "STOP_LOSS": "止损",
        "EMERGENCY_EXIT": "紧急退出",
        "ADMIN_HALT": "管理暂停",
        "SYSTEM": "系统异常",
        "DATA_SYNC": "行情同步",
        "ADMIN_CONFIG": "管理配置",
    }

    def _to_zh_key(self, k: str) -> str:
        return self._KEY_ZH_MAP.get(k, k)

    def _to_zh_value(self, k: str, v: Any) -> Any:
        """针对常见字段做值的中文化（只影响 Telegram 展示）"""
        if v is None:
            return v

        if k == "level":
            return self._LEVEL_ZH_MAP.get(str(v).upper(), v)

        if k == "event":
            return self._EVENT_ZH_MAP.get(str(v), v)

        if k == "side":
            return self._SIDE_ZH_MAP.get(str(v).upper(), v)

        if k == "reason_code":
            return self._REASON_CODE_ZH_MAP.get(str(v), v)

        # 交易对 symbol / 交易所 exchange：按你的要求保持默认值，不翻译
        return v

    def _translate_obj(self, obj: Any) -> Any:
        """递归把 dict 的 key（以及部分字段的 value）翻译成中文用于 Telegram 展示"""
        if isinstance(obj, dict):
            new: Dict[str, Any] = {}
            for k, v in obj.items():
                kk = self._to_zh_key(str(k))
                # 只有原始 key 才能决定 value 的翻译规则
                vv = self._to_zh_value(str(k), self._translate_obj(v))
                new[kk] = vv
            return new
        if isinstance(obj, list):
            return [self._translate_obj(x) for x in obj]
        return obj

    def send_alert_zh(
        self,
        *,
        title: str,
        summary_kv: Dict[str, Any],
        payload: Dict[str, Any],
        json_indent: int = 2,
    ) -> None:
        """发送到 Telegram 的内容全部中文化（仅展示层）。

        - title：你传入的标题（建议中文）
        - summary_kv：摘要键值（会翻译 key，并翻译部分 value，如 level/event/side/reason_code）
        - payload：原始英文 payload（这里会翻译成中文后再展示在 Telegram 的 JSON 代码块里）
        """
        # 先翻译 summary 和 payload（只用于展示）
        summary_zh: Dict[str, Any] = self._translate_obj(summary_kv)
        payload_zh: Dict[str, Any] = self._translate_obj(payload)

        # 摘要行：key=value
        summary_lines = [f"{k}={v}" for k, v in summary_zh.items()]

        self.send_alert(title=title, summary_lines=summary_lines, payload=payload_zh, json_indent=json_indent)

    # 兼容旧调用
    def send(self, text: str) -> None:
        self.send_text(text)
