from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MemoryTier:
    name: str
    capacity_gb: float
    bandwidth_gbps: float
    latency_ms: float = 0.0
    utilization: float = 0.0


class MemoryHierarchy:
    """Simple explicit memory hierarchy used by the simulator."""

    def __init__(self, config: Dict[str, Dict[str, float]] | None = None):
        base = {
            "HBM": {"capacity_gb": 192.0, "bandwidth_gbps": 3350.0, "latency_ms": 0.1},
            "DRAM": {"capacity_gb": 1024.0, "bandwidth_gbps": 160.0, "latency_ms": 0.5},
            "CXL": {"capacity_gb": 2048.0, "bandwidth_gbps": 64.0, "latency_ms": 1.0},
            "NVMe": {"capacity_gb": 20000.0, "bandwidth_gbps": 12.0, "latency_ms": 2.0},
            "QPU": {"capacity_gb": 64.0, "bandwidth_gbps": 2000.0, "latency_ms": 0.05},
        }
        if config:
            base.update(config)

        self.tiers: Dict[str, MemoryTier] = {
            name: MemoryTier(name=name, **params)
            for name, params in base.items()
        }

    def bandwidth_for(self, tier_name: str) -> float:
        return self.tiers[tier_name].bandwidth_gbps

    def transfer_time(self, bytes_to_move: float, src: str, dst: str, congestion: float = 1.0) -> float:
        src_tier = self.tiers.get(src)
        dst_tier = self.tiers.get(dst)
        if src_tier is None:
            raise KeyError(f"Source tier '{src}' is not present in memory hierarchy.")
        if dst_tier is None:
            raise KeyError(f"Destination tier '{dst}' is not present in memory hierarchy.")
        bandwidth = min(src_tier.bandwidth_gbps, dst_tier.bandwidth_gbps)
        gb = bytes_to_move / (1024 ** 3)
        seconds = gb / (bandwidth / 8.0) * congestion
        return seconds

    def utilization_snapshot(self) -> Dict[str, float]:
        return {name: tier.utilization for name, tier in self.tiers.items()}

    def update_utilization(self, tier_name: str, value: float) -> None:
        self.tiers[tier_name].utilization = max(0.0, min(1.0, value))

    def __repr__(self) -> str:
        return f"MemoryHierarchy({list(self.tiers.keys())})"
