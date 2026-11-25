"""Tests for the properties the first version of the simulator violated (see docs/MODEL.md)."""
import os
import subprocess
import sys

import pytest

from src.config import SimulationConfig
from src.simulator import CONTROLLER_MAP, compare_controllers, run_controlled_experiment, simulate_run

ALL = ["reactive", "predictive", "wsa", "adaptive", "confidence_gated_adaptive", "oracle"]


def _run(controller, workload="agentic", precision="int4", seed=7, n=24, **kw):
    return simulate_run(SimulationConfig(workload=workload, precision=precision, controller=controller, num_ops=n, seed=seed, **kw))


@pytest.mark.parametrize("controller", ALL)
@pytest.mark.parametrize("workload", ["agentic", "long-running", "static-llm"])
def test_regret_is_never_negative(controller, workload):
    assert all(r["regret"] >= 0.0 for r in _run(controller, workload).records)


@pytest.mark.parametrize("workload", ["agentic", "branching", "tool-heavy"])
def test_oracle_has_zero_regret(workload):
    # Costs are only comparable inside one run (the workload state evolves differently per controller), so the lower-bound
    # property is checked per run: regret = cost - minimum >= 0 for everyone (previous test) and exactly 0 for the oracle.
    oracle = _run("oracle", workload)
    assert max(r["regret"] for r in oracle.records) < 1e-9
    assert all(r["selected_resource"] == r["oracle_resource"] for r in oracle.records)


def test_reactive_is_data_local_and_not_the_oracle():
    run = _run("reactive")
    assert all(r["selected_resource"] == r["home_resource"] and r["transfer_time"] == 0.0 for r in run.records)
    assert run.to_summary()["oracle_match_pct"] < 1.0  # the oracle sometimes moves data, so Reactive is not the oracle


def test_prediction_cost_is_charged_only_to_prediction_controllers():
    for controller in ("reactive", "wsa", "oracle"):
        assert not CONTROLLER_MAP[controller].uses_prediction
    for r in _run("predictive").records:
        assert r["controller_time"] == pytest.approx(0.025 + r["prediction_cost"])
    for r in _run("reactive").records:
        assert r["controller_time"] == pytest.approx(0.010)


def test_seed_changes_the_workload_and_results_are_reproducible():
    a, b = _run("reactive", seed=7).to_summary(), _run("reactive", seed=11).to_summary()
    assert a["mean_end_to_end_ms"] != b["mean_end_to_end_ms"]
    assert _run("adaptive", seed=7).to_summary() == _run("adaptive", seed=7).to_summary()


def test_results_do_not_depend_on_python_hash_randomisation():
    code = ("from src.config import SimulationConfig; from src.simulator import simulate_run; "
            "print(simulate_run(SimulationConfig(workload='agentic', precision='int4', controller='adaptive', num_ops=24, seed=7)).to_summary()['mean_end_to_end_ms'])")
    outs = set()
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout.strip())
    assert len(outs) == 1


def test_memory_cost_enters_execution_time():
    fp16, int4 = _run("reactive", precision="fp16").records, _run("reactive", precision="int4").records
    assert all(r["execution_time"] >= r["memory_time"] for r in fp16)
    assert sum(r["memory_time"] for r in fp16) > sum(r["memory_time"] for r in int4)  # lower precision reads fewer bytes


def test_overlap_makes_execution_the_max_instead_of_the_sum():
    add = _run("reactive").records
    over = _run("reactive", overlap_compute_memory=True).records
    assert all(o["execution_time"] <= a["execution_time"] + 1e-12 for o, a in zip(over, add))


def test_calibration_is_measured_and_can_be_broken():
    def ece(offset):
        runs = compare_controllers(["agentic", "branching", "tool-heavy", "long-running"], ["predictive"], ["int4"], num_ops=36,
                                   seeds=[1, 2, 3, 4, 5], calibration_offset=offset)
        return sum(r.to_summary()["calibration_error"] for r in runs) / len(runs)
    assert ece(0.0) < 0.12
    assert ece(-0.3) > ece(0.0) + 0.15


def test_size_scale_scales_memory_time_not_compute_time():
    base, half = _run("reactive").records, _run("reactive", size_scale=0.5).records
    assert sum(r["compute_time"] for r in base) == pytest.approx(sum(r["compute_time"] for r in half))
    assert sum(r["memory_time"] for r in half) < sum(r["memory_time"] for r in base)
