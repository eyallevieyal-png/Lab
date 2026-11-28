"""Executable check that each of the 8 defects found in the first simulator is absent from the CURRENT execution path.

Run from the repository root:  python -m research.validation.verify_simulator_fixes
Exit code 0 only if all 8 checks pass. Writes research/validation/simulator_fixes_report.json.
Each check exercises the code that experiments actually import (src.simulator, src.workloads, src.memory), not a copy.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from statistics import mean

from src import simulator
from src.config import SimulationConfig
from src.memory import MemoryHierarchy
from src.simulator import ReactiveController, simulate_run
from src.workloads import load_workload

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fingerprint(hashseed: str) -> str:
    code = "import json;from src.workloads import load_workload;print(json.dumps(load_workload('agentic',24,seed=7),sort_keys=True))"
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout


def d1_stable_seeding():
    a, b = _fingerprint("1"), _fingerprint("12345")
    return a == b, "workload stream identical under two different PYTHONHASHSEED values"


def d2_reactive_not_oracle():
    src = inspect.getsource(ReactiveController.select_resource)
    rows = [r for s in (7, 11) for r in simulate_run(SimulationConfig(workload="agentic", controller="reactive", num_ops=24, seed=s)).records]
    differs = sum(1 for r in rows if r["selected_resource"] != r["oracle_resource"])
    return "oracle" not in src.lower() and differs > 0, f"Reactive source does not reference the oracle; it differs from the oracle on {differs}/{len(rows)} operations"


def d3_real_regret():
    oracle = simulate_run(SimulationConfig(workload="agentic", controller="oracle", num_ops=24, seed=7)).records
    others = [r for c in ("reactive", "predictive", "adaptive") for r in simulate_run(SimulationConfig(workload="agentic", controller=c, num_ops=24, seed=7)).records]
    ok = all(abs(r["regret"]) < 1e-9 for r in oracle) and all(r["regret"] >= -1e-9 for r in others)
    return ok, f"oracle regret is 0 on {len(oracle)} ops; regret >= 0 on {len(others)} ops of 3 other controllers"


def d4_memory_cost_used():
    a = simulate_run(SimulationConfig(workload="agentic", precision="fp16", controller="reactive", num_ops=24, seed=7)).records
    b = simulate_run(SimulationConfig(workload="agentic", precision="int4", controller="reactive", num_ops=24, seed=7)).records
    ma, mb = sum(r["memory_time"] for r in a), sum(r["memory_time"] for r in b)
    return abs(ma - mb) > 1e-9, f"total memory time differs by precision (fp16 {ma:.3f} vs int4 {mb:.3f} ms)"


def d5_prediction_cost_charged():
    p = simulate_run(SimulationConfig(workload="agentic", controller="predictive", num_ops=24, seed=7)).records
    r = simulate_run(SimulationConfig(workload="agentic", controller="reactive", num_ops=24, seed=7)).records
    cp, cr = sum(x["prediction_cost_charged"] for x in p), sum(x["prediction_cost_charged"] for x in r)
    in_time = all(abs(x["controller_time"] - (0.025 + x["prediction_cost_charged"])) < 1e-9 for x in p)
    return cp > 0 and cr == 0 and in_time, (f"charged to Predictive ({cp:.3f} ms, included in its controller_time) and not to Reactive ({cr:.3f} ms). "
                                            "Note: the older field `prediction_cost` is reported for every controller even when not charged")


def d6_calibration_can_vary():
    ok_cal = simulate_run(SimulationConfig(workload="agentic", controller="predictive", num_ops=200, seed=7, calibration_offset=0.0)).to_summary()["calibration_error"]
    bad = simulate_run(SimulationConfig(workload="agentic", controller="predictive", num_ops=200, seed=7, calibration_offset=-0.3)).to_summary()["calibration_error"]
    return bad > ok_cal + 0.05, f"expected calibration error {ok_cal:.3f} (offset 0) vs {bad:.3f} (offset -0.3, overconfident)"


def d7_seed_changes_workload():
    a, b = load_workload("agentic", 24, seed=1), load_workload("agentic", 24, seed=2)
    return a != b, "different seeds give different operation streams"


def d8_bandwidth_units():
    h = MemoryHierarchy()
    t = h.transfer_time(1024 ** 3, "HBM", "HBM", congestion=1.0)   # 1 GiB over 3350 GB/s
    ops = load_workload("agentic", 24, seed=7)
    typical_ms = mean(h.transfer_time(o["size_bytes"], "DRAM", "HBM") * 1000 for o in ops)
    return abs(t - 1 / 3350) < 1e-9 and typical_ms > 0.1, f"1 GiB over HBM = {t * 1000:.3f} ms (no /8); typical DRAM->HBM transfer {typical_ms:.2f} ms (not ~0)"


CHECKS = [
    ("D1", "workloads seeded with per-process hash()", d1_stable_seeding),
    ("D2", "Reactive returned the oracle's resource", d2_reactive_not_oracle),
    ("D3", "regret was a fixed fraction, oracle had positive regret", d3_real_regret),
    ("D4", "memory cost computed but never used", d4_memory_cost_used),
    ("D5", "prediction cost never charged", d5_prediction_cost_charged),
    ("D6", "prediction correct with probability == confidence (no miscalibration possible)", d6_calibration_can_vary),
    ("D7", "seed did not change the workload", d7_seed_changes_workload),
    ("D8", "bandwidth divided by 8 / sizes so small transfers cost ~0", d8_bandwidth_units),
]


def main() -> int:
    results = []
    for cid, title, fn in CHECKS:
        ok, evidence = fn()
        results.append({"id": cid, "defect": title, "fixed_in_current_path": bool(ok), "evidence": evidence})
        print(f"{'PASS' if ok else 'FAIL'}  {cid}  {title}\n      {evidence}")
    out = os.path.join(ROOT, "research", "validation", "simulator_fixes_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"module": "src.simulator", "checks": results, "all_fixed": all(r["fixed_in_current_path"] for r in results)}, fh, indent=2, ensure_ascii=False)
    return 0 if all(r["fixed_in_current_path"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
