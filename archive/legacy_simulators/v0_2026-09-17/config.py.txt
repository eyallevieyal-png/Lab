from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class MemoryTierConfig:
    capacity_gb: float
    bandwidth_gbps: float
    latency_ms: float = 0.0


@dataclass
class PrecisionConfig:
    name: str
    bits: int
    compute_multiplier: float
    memory_multiplier: float
    arithmetic_intensity: float


@dataclass
class HardwareConfig:
    memory: Dict[str, MemoryTierConfig] = field(
        default_factory=lambda: {
            "HBM": MemoryTierConfig(capacity_gb=192.0, bandwidth_gbps=3350.0, latency_ms=0.1),
            "DRAM": MemoryTierConfig(capacity_gb=1024.0, bandwidth_gbps=160.0, latency_ms=0.5),
            "CXL": MemoryTierConfig(capacity_gb=2048.0, bandwidth_gbps=64.0, latency_ms=1.0),
            "NVMe": MemoryTierConfig(capacity_gb=20000.0, bandwidth_gbps=12.0, latency_ms=2.0),
        }
    )
    qpu_speed_ratio: float = 0.75
    qpu_queue_penalty: float = 0.35
    controller_overhead: Dict[str, float] = field(
        default_factory=lambda: {
            "Reactive": 0.010,
            "Predictive": 0.025,
            "WSA": 0.035,
            "Adaptive": 0.040,
            "Oracle": 0.012,
            "Confidence-Gated Adaptive": 0.045,
        }
    )
    prediction_cost: float = 0.015
    transfer_congestion_factor: float = 1.15


@dataclass
class SimulationConfig:
    workload: str = "agentic"
    precision: str = "int4"
    controller: str = "adaptive"
    num_ops: int = 64
    seed: int = 7
    speculative_acceptance: float = 0.75
    speculative_enabled: bool = True
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


PRECISION_CATALOG: Dict[str, PrecisionConfig] = {
    "fp16": PrecisionConfig("fp16", 16, 1.0, 1.0, 0.85),
    "int4": PrecisionConfig("int4", 4, 1.6, 0.4, 1.4),
    "bitnet_b1.58": PrecisionConfig("bitnet_b1.58", 1, 1.9, 0.32, 1.8),
    "bitnet_a4.8": PrecisionConfig("bitnet_a4.8", 4, 1.7, 0.35, 1.6),
}


def get_precision(name: str) -> PrecisionConfig:
    key = name.lower().replace(" ", "")
    try:
        return PRECISION_CATALOG[key]
    except KeyError as exc:  # pragma: no cover - simple user error path
        valid = ", ".join(sorted(PRECISION_CATALOG))
        raise ValueError(f"Unsupported precision '{name}'. Available: {valid}") from exc


DEFAULT_SIMULATION_CONFIG = SimulationConfig()
