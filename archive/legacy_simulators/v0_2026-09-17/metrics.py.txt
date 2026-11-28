from __future__ import annotations

from typing import Iterable


def quantile(values: Iterable[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    idx = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
    return float(xs[idx])
