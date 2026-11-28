from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Any, Dict, Iterable, List

from src.config import DEFAULT_SIMULATION_CONFIG, SimulationConfig, get_precision
from src.memory import MemoryHierarchy
from src.workloads import load_workload


RESOURCE_SPECS = {
    "HBM": {"compute_ratio": 1.0, "latency_ms": 0.02, "queue_cost": 0.03},
    "DRAM": {"compute_ratio": 0.82, "latency_ms": 0.035, "queue_cost": 0.04},
    "CXL": {"compute_ratio": 0.65, "latency_ms": 0.05, "queue_cost": 0.07},
    "NVMe": {"compute_ratio": 0.48, "latency_ms": 0.08, "queue_cost": 0.09},
    "QPU": {"compute_ratio": 1.35, "latency_ms": 0.01, "queue_cost": 0.12},
}


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
    def __init__(self, seed: int = 0):
        self.seed = seed

    def predict(self, workload_name: str, op: Dict[str, Any], current_state: Dict[str, Any]) -> PredictionResult:
        actual = op["oracle_resource"]
        rng = random.Random(self.seed + op["operation_id"] + len(workload_name))
        confidence = min(0.99, max(0.12, 0.35 + rng.random() * 0.6 + current_state.get("workload_state", 0.0) * 0.15))
        chance = rng.random()
        if chance < confidence:
            predicted = actual
            correct = True
        else:
            alternatives = [r for r in RESOURCE_SPECS if r != actual]
            predicted = alternatives[rng.randrange(len(alternatives))]
            correct = False
        top_k = [actual, predicted]
        cost = 0.003 + rng.random() * 0.015
        return PredictionResult(
            correct=correct,
            confidence=float(confidence),
            accuracy=float(1.0 if correct else 0.0),
            predicted_resource=predicted,
            actual_resource=actual,
            cost=cost,
            top_k=top_k,
        )


class BaseController:
    name: str = "base"
    overhead: float = 0.0

    def select_resource(self, op: Dict[str, Any], state: Dict[str, Any], prediction: PredictionResult | None = None) -> str:
        raise NotImplementedError


class ReactiveController(BaseController):
    name = "Reactive"
    overhead = 0.010

    def select_resource(self, op: Dict[str, Any], state: Dict[str, Any], prediction: PredictionResult | None = None) -> str:
        return op["oracle_resource"]


class PredictiveController(BaseController):
    name = "Predictive"
    overhead = 0.025

    def select_resource(self, op: Dict[str, Any], state: Dict[str, Any], prediction: PredictionResult | None = None) -> str:
        return prediction.predicted_resource if prediction else op["oracle_resource"]


class WorkloadStateAwareController(BaseController):
    name = "WSA"
    overhead = 0.035

    def select_resource(self, op: Dict[str, Any], state: Dict[str, Any], prediction: PredictionResult | None = None) -> str:
        workload_state = state.get("workload_state", 0.5)
        if workload_state > 0.7 and op["needs_sync"]:
            return "QPU" if op["tool_call"] else "HBM"
        if op["operation_type"] == "tool_call":
            return "DRAM"
        return op["oracle_resource"]


class AdaptiveController(BaseController):
    name = "Adaptive"
    overhead = 0.040

    def select_resource(self, op: Dict[str, Any], state: Dict[str, Any], prediction: PredictionResult | None = None) -> str:
        if prediction is None:
            return op["oracle_resource"]
        expected_gain = max(0.0, prediction.confidence - 0.45) * 0.25
        if prediction.cost + expected_gain < 0.06:
            return prediction.predicted_resource
        return op["oracle_resource"]


class OracleController(BaseController):
    name = "Oracle"
    overhead = 0.012

    def select_resource(self, op: Dict[str, Any], state: Dict[str, Any], prediction: PredictionResult | None = None) -> str:
        return op["oracle_resource"]


class ConfidenceGatedAdaptiveController(BaseController):
    name = "Confidence-Gated Adaptive"
    overhead = 0.045

    def select_resource(self, op: Dict[str, Any], state: Dict[str, Any], prediction: PredictionResult | None = None) -> str:
        if prediction is None:
            return op["oracle_resource"]
        threshold = 0.68 if op["needs_sync"] else 0.62
        cost_benefit = prediction.confidence - prediction.cost * 8.0
        if prediction.confidence >= threshold and cost_benefit > 0.25:
            return prediction.predicted_resource
        return op["oracle_resource"]


CONTROLLER_MAP = {
    "reactive": ReactiveController,
    "predictive": PredictiveController,
    "wsa": WorkloadStateAwareController,
    "adaptive": AdaptiveController,
    "oracle": OracleController,
    "confidence-gated adaptive": ConfidenceGatedAdaptiveController,
    "confidence_gated_adaptive": ConfidenceGatedAdaptiveController,
}


def _resource_speed(resource: str, precision: str) -> float:
    base = RESOURCE_SPECS[resource]["compute_ratio"]
    if precision == "int4":
        base *= 1.8
    elif precision == "bitnet_b1.58":
        base *= 2.4
    elif precision == "bitnet_a4.8":
        base *= 2.1
    return base


def _build_controller(name: str) -> BaseController:
    controller_cls = CONTROLLER_MAP.get(name.lower())
    if controller_cls is None:
        valid = ", ".join(sorted(CONTROLLER_MAP))
        raise ValueError(f"Unsupported controller '{name}'. Available: {valid}")
    return controller_cls()


@dataclass
class SimulationRun:
    workload: str
    controller: str
    precision: str
    seed: int
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        end_to_end = [r["end_to_end_time"] for r in self.records]
        useful = sum(1 for r in self.records if r["useful_placement"])
        harmful = sum(1 for r in self.records if r["harmful_placement"])
        total = max(1, len(self.records))
        regret = [r["regret"] for r in self.records]
        return {
            "workload": self.workload,
            "controller": self.controller,
            "precision": self.precision,
            "seed": self.seed,
            "p50": quantile(end_to_end, 0.50),
            "p95": quantile(end_to_end, 0.95),
            "p99": quantile(end_to_end, 0.99),
            "throughput_ops_per_sec": len(self.records) / max(1e-6, sum(end_to_end) / 1.0),
            "useful_pct": useful / total,
            "harmful_pct": harmful / total,
            "mean_regret": sum(regret) / len(regret),
            "median_regret": median(regret),
        }


def quantile(values: Iterable[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    idx = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
    return float(xs[idx])


def simulate_run(config: SimulationConfig) -> SimulationRun:
    precision = get_precision(config.precision)
    controller = _build_controller(config.controller)
    hierarchy = MemoryHierarchy()
    workload_ops = load_workload(config.workload, config.num_ops)
    predictor = PredictionModel(config.seed)
    state = {"workload_state": 0.55, "prediction_history": []}
    records: List[Dict[str, Any]] = []

    for op in workload_ops:
        op = dict(op)
        op["precision"] = config.precision
        op["speculative_enabled"] = config.speculative_enabled
        op["speculative_acceptance"] = config.speculative_acceptance
        op["compute_cost"] = op["arithmetic_intensity"] * precision.compute_multiplier
        op["memory_cost"] = op["size_bytes"] / (1024 ** 2) * precision.memory_multiplier
        prediction = predictor.predict(config.workload, op, state)
        selected_resource = controller.select_resource(op, state, prediction)
        oracle_resource = op["oracle_resource"]
        queue_time = RESOURCE_SPECS.get(selected_resource, RESOURCE_SPECS["HBM"])["queue_cost"] * (0.5 + state.get("workload_state", 0.4))
        transfer_bytes = max(0.0, op["size_bytes"] * (0.35 if selected_resource != oracle_resource else 0.12))
        transfer_time = hierarchy.transfer_time(transfer_bytes, "HBM" if selected_resource == "HBM" else "DRAM", selected_resource, congestion=1.1 + state.get("workload_state", 0.0) * 0.3)
        if selected_resource == "QPU":
            prep_time = 0.05
            measurement_time = 0.02
            post_processing = 0.02
            synchronization = 0.08
        else:
            prep_time = 0.01 if op["operation_type"] == "tool_call" else 0.005
            measurement_time = 0.0
            post_processing = 0.0
            synchronization = 0.02 if op["needs_sync"] else 0.0
        execution_time = (op["compute_cost"] / max(0.15, _resource_speed(selected_resource, config.precision))) * 0.18
        if selected_resource == "QPU":
            execution_time *= 0.4
        controller_time = controller.overhead + prediction.cost if hasattr(controller, "predict") else controller.overhead
        end_to_end_time = queue_time + transfer_time + prep_time + execution_time + measurement_time + post_processing + synchronization + controller_time
        oracle_time = queue_time * 0.5 + max(0.001, transfer_time * 0.4) + prep_time * 0.5 + execution_time * 0.8 + synchronization * 0.4
        useful = selected_resource == oracle_resource
        harmful = not useful
        regret = max(0.0, end_to_end_time - oracle_time)
        record = {
            "run_id": f"{config.seed}-{config.workload}-{config.controller}",
            "workload": config.workload,
            "operation_id": op["operation_id"],
            "operation_type": op["operation_type"],
            "selected_resource": selected_resource,
            "oracle_resource": oracle_resource,
            "placement": selected_resource,
            "predicted_placement": prediction.predicted_resource,
            "prediction_confidence": prediction.confidence,
            "prediction_accuracy": prediction.accuracy,
            "queue_time": queue_time,
            "preparation_time": prep_time,
            "transfer_time": transfer_time,
            "execution_time": execution_time,
            "measurement_time": measurement_time,
            "post_processing_time": post_processing,
            "synchronization_time": synchronization,
            "controller_time": controller_time,
            "end_to_end_time": end_to_end_time,
            "useful_placement": useful,
            "harmful_placement": harmful,
            "regret": regret,
            "prediction_cost": prediction.cost,
            "reason_code": "REACTIVE_PREFERRED" if useful else "HIGH_TRANSFER_COST",
        }
        if prediction.confidence > 0.75 and selected_resource != oracle_resource:
            record["reason_code"] = "HIGH_EXPECTED_PREDICTION_GAIN"
        if selected_resource == "QPU" and op["needs_sync"]:
            record["reason_code"] = "HIGH_QUEUE_PRESSURE"
        records.append(record)
        state["workload_state"] = max(0.1, min(0.95, state["workload_state"] + (0.05 if useful else -0.04)))
        state["prediction_history"].append(prediction.confidence)

    return SimulationRun(
        workload=config.workload,
        controller=controller.name,
        precision=config.precision,
        seed=config.seed,
        records=records,
    )


def run_controlled_experiment(configs: List[SimulationConfig]) -> List[SimulationRun]:
    runs: List[SimulationRun] = []
    for config in configs:
        runs.append(simulate_run(config))
    return runs


def compare_controllers(
    workloads: List[str] | None = None,
    controllers: List[str] | None = None,
    precisions: List[str] | None = None,
    num_ops: int = 32,
    seeds: List[int] | None = None,
) -> List[SimulationRun]:
    workloads = workloads or ["agentic", "branching", "conversational", "long-running", "static-llm", "tool-heavy"]
    controllers = controllers or ["reactive", "predictive", "wsa", "adaptive", "confidence_gated_adaptive", "oracle"]
    precisions = precisions or ["fp16", "int4", "bitnet_b1.58"]
    seeds = seeds or [7, 11, 42]

    configs: List[SimulationConfig] = []
    for workload in workloads:
        for controller in controllers:
            for precision in precisions:
                for seed in seeds:
                    configs.append(
                        SimulationConfig(
                            workload=workload,
                            precision=precision,
                            controller=controller,
                            num_ops=num_ops,
                            seed=seed,
                            speculative_enabled=True,
                            speculative_acceptance=0.75,
                        )
                    )
    return run_controlled_experiment(configs)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    directory = path.rsplit("/", 1)[0] if "/" in path else "."
    if directory and directory != ".":
        import os
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
        "runs": [
            {"workload": run.workload, "controller": run.controller, "records": run.records}
            for run in runs
        ],
    }
    _save_json(output_path, payload)
    return payload


def generate_research_report(output_path: str = "experiments/results/research_report.json") -> Dict[str, Any]:
    runs = compare_controllers(workloads=["agentic", "branching"], controllers=["reactive", "predictive", "wsa", "adaptive", "confidence_gated_adaptive", "oracle"], precisions=["int4"], num_ops=24, seeds=[7, 11])
    summary = [run.to_summary() for run in runs]
    payload = {
        "experiment": "controlled-comparison",
        "summary": summary,
        "controllers": sorted({row["controller"] for row in summary}),
        "workloads": sorted({row["workload"] for row in summary}),
        "notes": [
            "Historical numbers are used as reference only and are not hard-coded into the simulator.",
            "This benchmark measures modeled end-to-end cost and oracle regret, not physical hardware results.",
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
