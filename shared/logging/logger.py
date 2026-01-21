
"""Structured logging helper."""

from __future__ import annotations
import logging
from typing import Any

class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            base.update(extra)
        return " ".join([f"{k}={repr(v)}" for k, v in base.items()])

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level.upper())
    import sys
    h = logging.StreamHandler(sys.stderr)  # 明确输出到 stderr
    h.setFormatter(KVFormatter())
    logger.addHandler(h)
    logger.propagate = False
    return logger

def log(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra={"extra": fields})
