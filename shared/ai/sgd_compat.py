from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

@dataclass
class SGDClassifierCompat:
    """轻量版 SGDClassifier（log_loss），用于满足 V8.3 文档对“SGDClassifier 增量学习”的口径。

    说明：
    - 不依赖 sklearn（便于容器最小化部署）
    - 支持 partial_fit / predict_proba / to_dict / from_dict
    - 仅支持二分类：y ∈ {0,1}
    """
    dim: int
    lr: float = 0.05
    l2: float = 1e-4
    bias: float = 0.0
    w: Optional[List[float]] = None
    seen: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = [0.0] * int(self.dim)

    def predict_proba(self, x: Sequence[float]) -> List[float]:
        z = self.bias
        for i, xi in enumerate(x):
            if i >= self.dim:
                break
            z += (self.w[i] or 0.0) * float(xi)
        p1 = _sigmoid(z)
        return [1.0 - p1, p1]

    def partial_fit(self, x: Sequence[float], y: int) -> None:
        y = 1 if int(y) == 1 else 0
        proba = self.predict_proba(x)[1]
        # gradient for log loss: (p - y) * x
        err = (proba - y)
        for i, xi in enumerate(x):
            if i >= self.dim:
                break
            wi = self.w[i]
            grad = err * float(xi) + self.l2 * wi
            self.w[i] = wi - self.lr * grad
        # bias
        self.bias -= self.lr * err
        self.seen += 1
        self.version += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "impl": "sgd_compat",
            "dim": int(self.dim),
            "lr": float(self.lr),
            "l2": float(self.l2),
            "bias": float(self.bias),
            "w": list(self.w or []),
            "seen": int(self.seen),
            "version": int(self.version),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], *, fallback_dim: int) -> "SGDClassifierCompat":
        dim = int(d.get("dim") or fallback_dim)
        return SGDClassifierCompat(
            dim=dim,
            lr=float(d.get("lr") or 0.05),
            l2=float(d.get("l2") or 1e-4),
            bias=float(d.get("bias") or 0.0),
            w=list(d.get("w") or [0.0] * dim),
            seen=int(d.get("seen") or 0),
            version=int(d.get("version") or 1),
        )
