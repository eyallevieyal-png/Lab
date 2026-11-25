"""Memory Frontier Labs simulator.

Model summary (all times in milliseconds; see docs/MODEL.md for the assumptions and what was changed from the first version):

* Every operation has a `home_resource`: the tier where its data lives. Running on another resource first moves the data.
* The cost of running an operation on a resource is `placement_cost(op, resource)`: queue + data transfer + preparation +
  execution (compute plus reading the operands from the tier) + measurement + post-processing + synchronisation.
* The ORACLE is the resource with the minimum placement cost. Regret = placement cost of the chosen resource minus that minimum,
  so it is never negative and is exactly zero for the oracle.
* Controllers differ only in HOW they choose. The Reactive controller is data-local (it never moves data) and has no access to
  the oracle. Prediction controllers use a noisy predictor of the oracle resource, pay a prediction cost and a controller overhead.
* Nothing here is physical hardware evidence: it is a model with explicit, adjustable assumptions.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.config import SimulationConfig, get_precision
from src.memory import MemoryHierarchy
from src.workloads import load_workload

MS = 1000.0
EPS = 1e-9

RESOURCE_SPECS = {
    "HBM": {"compute_ratio": 1.0, "latency_ms": 0.02, "queue_cost": 0.03},
    "DRAM": {"compute_ratio": 0.82, "latency_ms": 0.035, "queue_cost": 0.04},
    "CXL": {"compute_ratio": 0.65, "latency_ms": 0.05, "queue_cost": 0.07},
    "NVMe": {"compute_ratio": 0.48, "latency_ms": 0.08, "queue_cost": 0.09},
    "QPU": {"compute_ratio": 1.35, "latency_ms": 0.01, "queue_cost": 0.12},
}
RESOURCES = list(RESOURCE_SPECS)


@dataclass
class PredictionResult:
    correct: bool
    confidence: float
    accuracy: float
    predicted_resource: str
    actual_resource: str
    cost: float
    top_k: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "confidence": self.confidence,
            "accuracy": self.accuracy,
            "predicted_resource": self.predicted_resource,
            "actual_resource": self.actual_resource,
            "cost": self.cost,
            "top_k": self.top_k,
        }


class PredictionModel:
    """A noisy predictor of the oracle resource. The probability of being right is the reported confidence plus
    `calibration_offset` (0 = calibrated by construction; negative = overconfident), so calibration can be varied and measured."""

    def __init__(self, seed: int = 0, calibration_offset: float = 0.0):
        self.seed = seed
        self.calibration_offset = calibration_offset

    def predict(self, workload_name: str, op: Dict[str, Any], current_state: Dict[str, Any], oracle_resource: str) -> PredictionResult:
        rng = random.Random(f"{self.seed}|{workload_name}|{op['operation_id']}")
        confidence = min(0.99, max(0.12, 0.35 + rng.random() * 0.6 + current_state.get("workload_state", 0.0) * 0.15))
        p_correct = min(1.0, max(0.0, confidence + self.calibration_offset))
        correct = rng.random() < p_correct
        if correct:
            predicted = oracle_resource
        else:
            alternatives = [r for r in RESOURCES if r != oracle_resource]
            predicted = alternatives[rng.randrange(len(alternatives))]
        cost = 0.003 + rng.random() * 0.015
        return PredictionResult(
            correct=correct,
            confidence=float(confidence),
            accuracy=float(1.0 if correct else 0.0),
            predicted_resource=predicted,
            actual_resource=oracle_resource,
            cost=cost,
            top_k=[oracle_resource, predicted],
        )


# ── cost model ───────────────────────────────────────────────

def _resource_speed(resource: str, precision: str) -> float:
    base = RESOURCE_SPECS[resource]["compute_ratio"]
    if precision == "int4":
        base *= 1.8
    elif precision == "bitnet_b1.58":
        base *= 2.4
    elif precision == "bitnet_a4.8":
        base *= 2.1
    return base


def placement_components(op: Dict[str, Any], resource: str, state: Dict[str, Any], hierarchy: MemoryHierarchy,
                         config: SimulationConfig) -> Dict[str, float]:
    """Every time component of running `op` on `resource` (milliseconds), without the controller's own overhead."""
    precision = get_precision(config.precision)
    workload_state = state.get("workload_state", 0.4)
    queue = RESOURCE_SPECS[resource]["queue_cost"] * (0.5 + workload_state)
    operand_bytes = op["size_bytes"] * precision.memory_multiplier
    home = op["home_resource"]
    if resource == home:
        transfer = 0.0
    else:
        congestion = 1.1 + workload_state * 0.3
        transfer = hierarchy.transfer_time(operand_bytes, home, resource, congestion) * MS + hierarchy.latency_ms_for(resource)
    memory = hierarchy.transfer_time(operand_bytes, resource, resource, 1.0) * MS + hierarchy.latency_ms_for(resource)
    compute = op["arithmetic_intensity"] * precision.compute_multiplier / max(0.15, _resource_speed(resource, config.precision)) * config.compute_ms_per_unit
    if resource == "QPU":
        compute *= 0.4
        prep, measurement, post, sync = 0.05, 0.02, 0.02, 0.08
    else:
        prep = 0.01 if op["operation_type"] == "tool_call" else 0.005
        measurement, post = 0.0, 0.0
        sync = 0.02 if op["needs_sync"] else 0.0
    execution = max(compute, memory) if config.overlap_compute_memory else compute + memory
    parts = {"queue": queue, "transfer": transfer, "preparation": prep, "compute": compute, "memory": memory,
             "execution": execution, "measurement": measurement, "post_processing": post, "synchronization": sync}
    parts["total"] = queue + transfer + prep + execution + measurement + post + sync
    return parts


# ── controllers ──────────────────────────────────────────────

@dataclass
class Decision:
    """What a controller may look at when choosing a resource."""
    op: Dict[str, Any]
    state: Dict[str, Any]
    prediction: Optional[PredictionResult]
    cost_of: Callable[[str], float]  # model evaluation of the placement cost of one resource


class BaseController:
    name = "base"
    overhead = 0.0
    uses_prediction = False

    def select_resource(self, d: Decision) -> str:
        raise NotImplementedError


class ReactiveController(BaseController):
    """Data-local: runs where the data is. It never moves data and has no access to the oracle."""
    name = "Reactive"
    overhead = 0.010

    def select_resource(self, d: Decision) -> str:
        return d.op["home_resource"]


class PredictiveController(BaseController):
    """Always trusts the predictor."""
    name = "Predictive"
    overhead = 0.025
    uses_prediction = True

    def select_resource(self, d: Decision) -> str:
        return d.prediction.predicted_resource


class WorkloadStateAwareController(BaseController):
    name = "WSA"
    overhead = 0.035

    def select_resource(self, d: Decision) -> str:
        workload_state = d.state.get("workload_state", 0.5)
        if workload_state > 0.7 and d.op["needs_sync"]:
            return "QPU" if d.op["tool_call"] else "HBM"
        if d.op["operation_type"] == "tool_call":
            return "DRAM"
        return d.op["home_resource"]


def expected_net_benefit(d: Decision) -> float:
    """p*B - (1-p)*L - c for acting on the prediction instead of staying data-local.
    B: model gain if the predicted resource is right. L: mean model loss over the alternatives the predictor could return wrongly."""
    pr = d.prediction
    home = d.op["home_resource"]
    base = d.cost_of(home)
    gain = max(0.0, base - d.cost_of(pr.predicted_resource))
    others = [r for r in RESOURCES if r != pr.predicted_resource]
    loss = mean(max(0.0, d.cost_of(r) - base) for r in others)
    p = pr.confidence
    return p * gain - (1.0 - p) * loss - pr.cost


class AdaptiveController(BaseController):
    """Acts on the prediction only when its expected net benefit is positive (otherwise stays data-local)."""
    name = "Adaptive"
    overhead = 0.040
    uses_prediction = True

    def select_resource(self, d: Decision) -> str:
        return d.prediction.predicted_resource if expected_net_benefit(d) > 0 else d.op["home_resource"]


class ConfidenceGatedAdaptiveController(BaseController):
    """Adaptive, and additionally requires the confidence to pass a gate (stricter when synchronisation is needed)."""
    name = "Confidence-Gated Adaptive"
    overhead = 0.045
    uses_prediction = True

    def select_resource(self, d: Decision) -> str:
        threshold = 0.68 if d.op["needs_sync"] else 0.62
        if d.prediction.confidence >= threshold and expected_net_benefit(d) > 0:
            return d.prediction.predicted_resource
        return d.op["home_resource"]


class OracleController(BaseController):
    """Idealised lower bound: the resource with the minimum placement cost, at no controller overhead."""
    name = "Oracle"
    overhead = 0.0

    def select_resource(self, d: Decision) -> str:
        home = d.op["home_resource"]
        best, best_cost = home, d.cost_of(home)
        for r in RESOURCES:
            c = d.cost_of(r)
            if c < best_cost - EPS:
                best, best_cost = r, c
        return best


CONTROLLER_MAP = {
    "reactive": ReactiveController,
    "predictive": PredictiveController,
    "wsa": WorkloadStateAwareController,
    "adaptive": AdaptiveController,
    "oracle": OracleController,
    "confidence-gated adaptive": ConfidenceGatedAdaptiveController,
    "confidence_gated_adaptive": ConfidenceGatedAdaptiveController,
}


def _build_controller(name: str) -> BaseController:
    controller_cls = CONTROLLER_MAP.get(name.lower())
    if controller_cls is None:
        valid = ", ".join(sorted(CONTROLLER_MAP))
        raise ValueError(f"Unsupported controller '{name}'. Available: {valid}")
    return controller_cls()


# ── metrics ──────────────────────────────────────────────────

def quantile(values: Iterable[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    idx = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
    return float(xs[idx])


def expected_calibration_error(confidences: List[float], correct: List[float], bins: int = 10) -> float:
    """Expected calibration error: the weighted gap between mean confidence and accuracy per confidence bin."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            total += len(idx) / n * abs(mean(confidences[i] for i in idx) - mean(correct[i] for i in idx))
    return total


@dataclass
class SimulationRun:
    workload: str
    controller: str
    precision: str
    seed: int
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        end_to_end = [r["end_to_end_time"] for r in self.records]
        total = max(1, len(self.records))
        regret = [r["regret"] for r in self.records]
        return {
            "workload": self.workload,
            "controller": self.controller,
            "precision": self.precision,
            "seed": self.seed,
            "mean_end_to_end_ms": mean(end_to_end) if end_to_end else 0.0,
            "p50": quantile(end_to_end, 0.50),
            "p95": quantile(end_to_end, 0.95),
            "p99": quantile(end_to_end, 0.99),
            "throughput_ops_per_sec": len(self.records) / max(1e-9, sum(end_to_end) / MS),
            "useful_pct": sum(1 for r in self.records if r["useful_placement"]) / total,
            "harmful_pct": sum(1 for r in self.records if r["harmful_placement"]) / total,
            "neutral_pct": sum(1 for r in self.records if not r["useful_placement"] and not r["harmful_placement"]) / total,
            "mean_regret": mean(regret) if regret else 0.0,
            "median_regret": median(regret) if regret else 0.0,
            "mean_total_regret": mean(r["total_regret"] for r in self.records) if self.records else 0.0,
            "oracle_match_pct": sum(1 for r in self.records if r["selected_resource"] == r["oracle_resource"]) / total,
            "prediction_accuracy": mean(r["prediction_accuracy"] for r in self.records) if self.records else 0.0,
            "calibration_error": expected_calibration_error([r["prediction_confidence"] for r in self.records],
                                                            [r["prediction_accuracy"] for r in self.records]),
            "memory_share_of_execution": (sum(r["memory_time"] for r in self.records) /
                                          max(1e-12, sum(r["memory_time"] + r["compute_time"] for r in self.records))),
        }


# ── simulation ───────────────────────────────────────────────

def simulate_run(config: SimulationConfig) -> SimulationRun:
    controller = _build_controller(config.controller)
    hierarchy = MemoryHierarchy()
    workload_ops = load_workload(config.workload, config.num_ops, seed=config.seed)
    predictor = PredictionModel(config.seed, config.calibration_offset)
    state: Dict[str, Any] = {"workload_state": 0.55}
    records: List[Dict[str, Any]] = []

    for op in workload_ops:
        op = dict(op)
        op["precision"] = config.precision
        components = {r: placement_components(op, r, state, hierarchy, config) for r in RESOURCES}
        cost_of = lambda r, c=components: c[r]["total"]  # noqa: E731
        home = op["home_resource"]
        min_cost = min(cost_of(r) for r in RESOURCES)
        oracle_resource = home if cost_of(home) <= min_cost + EPS else min(RESOURCES, key=cost_of)
        prediction = predictor.predict(config.workload, op, state, oracle_resource)
        decision = Decision(op=op, state=state, prediction=prediction, cost_of=cost_of)
        selected = controller.select_resource(decision)
        chosen = components[selected]
        prediction_charge = prediction.cost if controller.uses_prediction else 0.0
        controller_time = controller.overhead + prediction_charge
        end_to_end = chosen["total"] + controller_time
        delta = chosen["total"] - components[home]["total"]  # versus staying data-local
        useful, harmful = delta < -EPS, delta > EPS
        record = {
            "run_id": f"{config.seed}-{config.workload}-{controller.name}",
            "workload": config.workload,
            "operation_id": op["operation_id"],
            "operation_type": op["operation_type"],
            "home_resource": home,
            "selected_resource": selected,
            "oracle_resource": oracle_resource,
            "predicted_placement": prediction.predicted_resource,
            "prediction_confidence": prediction.confidence,
            "prediction_accuracy": prediction.accuracy,
            "prediction_cost": prediction.cost,
            "queue_time": chosen["queue"],
            "preparation_time": chosen["preparation"],
            "transfer_time": chosen["transfer"],
            "compute_time": chosen["compute"],
            "memory_time": chosen["memory"],
            "execution_time": chosen["execution"],
            "measurement_time": chosen["measurement"],
            "post_processing_time": chosen["post_processing"],
            "synchronization_time": chosen["synchronization"],
            "controller_time": controller_time,
            "placement_cost": chosen["total"],
            "end_to_end_time": end_to_end,
            "useful_placement": useful,
            "harmful_placement": harmful,
            "regret": chosen["total"] - min_cost,
            "total_regret": end_to_end - min_cost,
        }
        records.append(record)
        state["workload_state"] = max(0.1, min(0.95, state["workload_state"] + (0.05 if delta <= EPS else -0.04)))

    return SimulationRun(workload=config.workload, controller=controller.name, precision=config.precision, seed=config.seed, records=records)


def run_controlled_experiment(configs: List[SimulationConfig]) -> List[SimulationRun]:
    return [simulate_run(config) for config in configs]


def compare_controllers(
    workloads: List[str] | None = None,
    controllers: List[str] | None = None,
    precisions: List[str] | None = None,
    num_ops: int = 32,
    seeds: List[int] | None = None,
    calibration_offset: float = 0.0,
) -> List[SimulationRun]:
    workloads = workloads or ["agentic", "branching", "conversational", "long-running", "static-llm", "tool-heavy"]
    controllers = controllers or ["reactive", "predictive", "wsa", "adaptive", "confidence_gated_adaptive", "oracle"]
    precisions = precisions or ["fp16", "int4", "bitnet_b1.58"]
    seeds = seeds or [7, 11, 42]
    configs = [
        SimulationConfig(workload=w, precision=p, controller=c, num_ops=num_ops, seed=s, calibration_offset=calibration_offset)
        for w in workloads for c in controllers for p in precisions for s in seeds
    ]
    return run_controlled_experiment(configs)


def aggregate(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean and standard deviation across seeds for each workload x controller x precision."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in summary:
        groups.setdefault((row["workload"], row["controller"], row["precision"]), []).append(row)
    out = []
    for (w, c, p), rows in sorted(groups.items()):
        entry: Dict[str, Any] = {"workload": w, "controller": c, "precision": p, "seeds": len(rows)}
        for key in ("mean_end_to_end_ms", "mean_regret", "mean_total_regret", "useful_pct", "harmful_pct", "oracle_match_pct", "calibration_error"):
            vals = [r[key] for r in rows]
            entry[key] = mean(vals)
            entry[key + "_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(entry)
    return out


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def smoke_run(output_path: str = "experiments/results/smoke_summary.json") -> Dict[str, Any]:
    configs = [
        SimulationConfig(workload="agentic", precision="int4", controller="reactive", num_ops=24, seed=7),
        SimulationConfig(workload="agentic", precision="int4", controller="adaptive", num_ops=24, seed=7),
        SimulationConfig(workload="branching", precision="bitnet_b1.58", controller="predictive", num_ops=24, seed=7),
    ]
    runs = run_controlled_experiment(configs)
    payload = {
        "summary": [run.to_summary() for run in runs],
        "runs": [{"workload": run.workload, "controller": run.controller, "records": run.records} for run in runs],
    }
    _save_json(output_path, payload)
    return payload


def generate_research_report(output_path: str = "experiments/results/research_report.json") -> Dict[str, Any]:
    runs = compare_controllers(
        workloads=["agentic", "branching"],
        controllers=["reactive", "predictive", "wsa", "adaptive", "confidence_gated_adaptive", "oracle"],
        precisions=["int4"], num_ops=24, seeds=[7, 11, 42, 101, 202],
    )
    summary = [run.to_summary() for run in runs]
    payload = {
        "experiment": "controlled-comparison",
        "summary": summary,
        "aggregate": aggregate(summary),
        "controllers": sorted({row["controller"] for row in summary}),
        "workloads": sorted({row["workload"] for row in summary}),
        "notes": [
            "Historical numbers are used as reference only and are not hard-coded into the simulator.",
            "This benchmark measures modeled end-to-end cost and regret against the minimum-cost placement, not physical hardware results.",
            "Reactive is data-local and has no access to the oracle; the Oracle has zero regret by construction.",
        ],
    }
    _save_json(output_path, payload)
    return payload


if __name__ == "__main__":
    result = smoke_run()
    report = generate_research_report()
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print("\nResearch report saved to experiments/results/research_report.json")
    print(json.dumps({"controllers": report["controllers"], "workloads": report["workloads"]}, indent=2, sort_keys=True))
