"""Memory-tier abstraction and the backend interface.

`TierSpec` describes a tier. `TierBackend` is the interface the engine talks to: it answers "when will this read finish" and "how busy is this link".
`SimulatedBackend` implements it with one FIFO read channel per source tier (into HBM) plus optional background traffic from another tenant.
A real backend (CUDA host<->device copies, CXL/NUMA memory, NVMe reads) would implement the same three methods and be measured instead of modelled.

Every parameter below is ASSUMED, not measured. In particular CXL is a model only: nothing here has been run on CXL hardware.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Protocol


@dataclass(frozen=True)
class TierSpec:
    name: str
    capacity_blocks: Optional[int]   # None = unbounded
    read_gbps: float                 # sustained read bandwidth from this tier into HBM, GB/s (decimal)
    latency_ms: float                # fixed latency per transfer
    evidence: str                    # 'assumed' | 'spec' | 'measured'  (what the numbers rest on)


# HBM capacity is scaled down (64 blocks) so that a 48-session working set does not fit; the ratios between tiers are what matters.
DEFAULT_TIERS = (
    TierSpec("HBM", 64, 3350.0, 0.0, "spec"),
    TierSpec("DRAM", 192, 64.0, 0.001, "assumed"),      # PCIe Gen5 x16 class host link
    TierSpec("CXL", 512, 32.0, 0.002, "assumed"),       # MODEL ONLY: no CXL hardware was available or tested
    TierSpec("NVMe", None, 7.0, 0.1, "assumed"),
)
LOWER = ("DRAM", "CXL", "NVMe")   # order of the demotion waterfall


class TierBackend(Protocol):
    def reserve_read(self, src: str, nbytes: int, now: float) -> float:
        """Queue a read of `nbytes` from tier `src` into HBM at time `now` (ms); return its completion time."""

    def utilization(self, src: str, now: float, foreign_only: bool = False) -> float:
        """Fraction of the last window during which the `src` link was busy. `foreign_only=True` excludes the orchestrator's own transfers: this is the
        `u` of the H13 threshold (background load). A real backend gets it as total measured utilisation minus the transfers the orchestrator itself issued."""

    def queue_ms(self, src: str, now: float) -> float:
        """How long a new read would wait before it starts."""


class _Channel:
    def __init__(self, bg_util: float, bg_ms: float, rng: random.Random):
        self.free_at = 0.0
        self.busy: deque = deque()          # (start, end, issued_by_orchestrator) intervals inside the utilisation window
        self.bg_util, self.bg_ms, self.rng = bg_util, bg_ms, rng
        self.next_bg = rng.expovariate(bg_util / bg_ms) if bg_util > 0 else float("inf")

    def _advance_background(self, now: float) -> None:
        while self.next_bg <= now:
            start = max(self.free_at, self.next_bg)
            self.free_at = start + self.bg_ms
            self.busy.append((start, self.free_at, False))
            self.next_bg += self.rng.expovariate(self.bg_util / self.bg_ms)


class SimulatedBackend:
    WINDOW_MS = 200.0

    def __init__(self, tiers=DEFAULT_TIERS, bg_util: Optional[Dict[str, float]] = None, bg_ms: float = 4.0, seed: int = 0):
        self.spec: Dict[str, TierSpec] = {t.name: t for t in tiers}
        bg_util = bg_util or {}
        self.ch = {t.name: _Channel(bg_util.get(t.name, 0.0), bg_ms, random.Random(f"bg|{seed}|{t.name}")) for t in tiers if t.name != "HBM"}

    def transfer_ms(self, src: str, nbytes: int) -> float:
        s = self.spec[src]
        return s.latency_ms + nbytes / (s.read_gbps * 1e9) * 1000.0

    def reserve_read(self, src: str, nbytes: int, now: float) -> float:
        ch = self.ch[src]
        ch._advance_background(now)
        start = max(ch.free_at, now)
        ch.free_at = start + self.transfer_ms(src, nbytes)
        ch.busy.append((start, ch.free_at, True))
        return ch.free_at

    def utilization(self, src: str, now: float, foreign_only: bool = False) -> float:
        ch = self.ch[src]
        ch._advance_background(now)
        lo = now - self.WINDOW_MS
        while ch.busy and ch.busy[0][1] < lo:
            ch.busy.popleft()
        busy = sum(max(0.0, min(e, now) - max(s, lo)) for s, e, own in ch.busy if not (foreign_only and own))
        return min(1.0, busy / self.WINDOW_MS)

    def queue_ms(self, src: str, now: float) -> float:
        ch = self.ch[src]
        ch._advance_background(now)
        return max(0.0, ch.free_at - now)
