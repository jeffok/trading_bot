
from __future__ import annotations
from enum import Enum

class OrderEventType(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    ERROR = "ERROR"
    RECONCILED = "RECONCILED"

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class ReasonCode(str, Enum):
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    ADMIN_HALT = "ADMIN_HALT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"
    RECONCILE = "RECONCILE"
    DATA_SYNC = "DATA_SYNC"
    SYSTEM = "SYSTEM"
