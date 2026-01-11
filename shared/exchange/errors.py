from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class ExchangeError(Exception):
    """Base class for exchange-side errors."""
    pass


@dataclass
class RateLimitError(ExchangeError):
    """Raised when the exchange rate-limit is hit."""
    message: str = "rate limited"
    retry_after_seconds: Optional[float] = None
    group: Optional[str] = None
    severe: bool = False


class AuthError(ExchangeError):
    pass


class TemporaryError(ExchangeError):
    pass
