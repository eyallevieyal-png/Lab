"""Transfer-cost estimator: what a fetch would cost right now, from observed link state (queue) and the tier parameters."""
from __future__ import annotations

from .tiers import SimulatedBackend, TierBackend


class TransferCostEstimator:
    def __init__(self, backend: TierBackend):
        self.b = backend

    def fetch_ms(self, src: str, nbytes: int, now: float) -> float:
        """Expected time until `nbytes` from `src` are in HBM if requested now (queue wait + latency + streaming time)."""
        spec = self.b.spec[src]  # type: ignore[attr-defined]
        return self.b.queue_ms(src, now) + spec.latency_ms + nbytes / (spec.read_gbps * 1e9) * 1000.0

    def stream_ms(self, src: str, nbytes: int) -> float:
        """Transfer time on an idle link (the `t_x` of the H13 threshold)."""
        spec = self.b.spec[src]  # type: ignore[attr-defined]
        return spec.latency_ms + nbytes / (spec.read_gbps * 1e9) * 1000.0
