
"""Trace utilities."""

from __future__ import annotations
import secrets, time

def new_trace_id(prefix: str = "t") -> str:
    ms = int(time.time() * 1000)
    rand = secrets.token_hex(4)
    return f"{prefix}_{ms}_{rand}"
