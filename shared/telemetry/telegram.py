
from __future__ import annotations
import httpx

class Telegram:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()

    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, text: str) -> None:
        if not self.enabled():
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            with httpx.Client(timeout=10) as client:
                client.post(url, json={"chat_id": self.chat_id, "text": text})
        except Exception:
            pass
