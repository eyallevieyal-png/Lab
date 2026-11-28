from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class WorkloadSpec:
    name: str
    base_compute: float
    memory_pressure: float
    branching: float
    tool_ratio: float
    state_change_rate: float
    sequence_length: int

    def operation_profile(self, i: int, seed: int = 0) -> Dict[str, float | str | bool | int]:
        j = (i + 1) * (seed + 1)
        size_bytes = int(64 * 1024 * (1 + (i % 9) * 0.35) * (1.0 + self.memory_pressure))
        arithmetic_intensity = max(0.1, self.base_compute * (1.0 + (j % 5) * 0.08))
        tool_call = (i % max(2, int(10 * (1.0 - self.tool_ratio + 0.3))) == 0) if self.tool_ratio > 0 else False
        branching_factor = max(1, int(self.branching * (1 + (i % 4)))) if self.branching > 0 else 1
        needs_sync = (i % 4 == 0) or self.state_change_rate > 0.65
        oracle_resource = ["HBM", "DRAM", "CXL", "NVMe", "QPU"][i % 5]
        candidate = oracle_resource
        return {
            "operation_id": i,
            "operation_type": "inference" if not tool_call else "tool_call",
            "size_bytes": size_bytes,
            "arithmetic_intensity": arithmetic_intensity,
            "branching_factor": branching_factor,
            "tool_call": tool_call,
            "needs_sync": needs_sync,
            "oracle_resource": candidate,
            "workload_name": self.name,
        }


WORKLOAD_LIBRARY: Dict[str, WorkloadSpec] = {
    "agentic": WorkloadSpec("agentic", 1.3, 0.8, 0.9, 0.7, 0.8, 24),
    "branching": WorkloadSpec("branching", 1.1, 0.7, 1.6, 0.4, 0.6, 28),
    "conversational": WorkloadSpec("conversational", 0.8, 0.5, 0.2, 0.2, 0.3, 20),
    "long-running": WorkloadSpec("long-running", 0.9, 1.2, 0.5, 0.3, 0.9, 36),
    "static-llm": WorkloadSpec("static-llm", 0.7, 0.4, 0.1, 0.1, 0.2, 18),
    "tool-heavy": WorkloadSpec("tool-heavy", 1.5, 0.9, 0.4, 0.9, 0.7, 30),
}


def load_workload(name: str, num_ops: int | None = None) -> List[Dict[str, float | str | bool | int]]:
    workload = WORKLOAD_LIBRARY[name.lower()]
    count = num_ops if num_ops is not None else workload.sequence_length
    return [workload.operation_profile(i, seed=hash(name) % 17) for i in range(count)]
