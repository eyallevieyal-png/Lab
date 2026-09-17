import json

from src.simulator import SimulationConfig, run_controlled_experiment, smoke_run


def test_smoke_run_returns_summary():
    data = smoke_run("experiments/results/test_smoke.json")
    assert isinstance(data["summary"], list)
    assert len(data["summary"]) >= 3
    assert all("p50" in item for item in data["summary"])


def test_run_controlled_experiment_supports_multiple_controllers():
    configs = [
        SimulationConfig(workload="agentic", precision="int4", controller="reactive", num_ops=12, seed=1),
        SimulationConfig(workload="agentic", precision="int4", controller="adaptive", num_ops=12, seed=1),
        SimulationConfig(workload="branching", precision="bitnet_b1.58", controller="predictive", num_ops=12, seed=2),
    ]
    results = run_controlled_experiment(configs)
    controllers = {run.controller for run in results}
    assert controllers == {"Reactive", "Adaptive", "Predictive"}
    assert all(len(run.records) == 12 for run in results)


def test_reproducibility_with_same_seed_and_config():
    cfg = SimulationConfig(workload="conversational", precision="fp16", controller="reactive", num_ops=8, seed=42)
    first = run_controlled_experiment([cfg])[0]
    second = run_controlled_experiment([cfg])[0]
    assert first.to_summary() == second.to_summary()
