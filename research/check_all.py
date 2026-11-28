"""One command that answers "is everything in this repository still what it claims to be?".

    python -X utf8 -m research.check_all            (about 1.5 minutes, CPU only)
    python -X utf8 -m research.check_all --quick    (skips re-running the experiment scripts)

Checks: (1) the test suite; (2) the 8 simulator defects are absent from the current path; (3) results_summary.json regenerates to the same numbers; (4) each deterministic
experiment script regenerates its committed result byte for byte (a changed file is LEFT on disk so `git diff` shows what moved); (5) the committed registry matches its builder.
Exit code 0 only if all pass. Experiments that need external data or hardware are not re-run here (they are listed as skipped, not as passed).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETERMINISTIC = ["h6_calibrated", "h6_sensitivity", "h6_moe", "h13_kv_prefetch", "h13_sim_vs_formula", "h13_analytic_vs_sim"]
SKIPPED = {"h13_trace_background / h13_online_controller": "need the external BurstGPT CSV", "measure_gpu / measure_roofline": "hardware timings",
           "orchestrator.bench and the h16/h17 sweeps": "minutes each; run them explicitly"}


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(args: list[str]) -> int:
    return subprocess.run([sys.executable, "-X", "utf8", *args], cwd=ROOT).returncode


def main() -> int:
    quick = "--quick" in sys.argv
    results: list[tuple[str, bool]] = []
    results.append(("tests", subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode == 0))
    results.append(("8 simulator defects absent", run(["-m", "research.validation.verify_simulator_fixes"]) == 0))

    sys.path.insert(0, ROOT)
    from research.pipeline.make_results_summary import build
    committed = json.load(open(os.path.join(ROOT, "results_summary.json"), encoding="utf-8"))
    results.append(("results_summary.json regenerates", committed["aggregate"] == json.loads(json.dumps(build()["aggregate"]))))

    if not quick:
        for name in DETERMINISTIC:
            out = os.path.join(ROOT, "experiments", "results", name + ".json")
            before = sha(out)
            ok = run([os.path.join("experiments", name + ".py")]) == 0 and sha(out) == before
            results.append((f"{name} reproduces its committed result", ok))

    from research.registry.build_registry import REGISTRY
    reg = json.load(open(os.path.join(ROOT, "research", "EXPERIMENT_REGISTRY.json"), encoding="utf-8"))
    results.append(("registry up to date", {e["id"] for e in reg["experiments"]} == {e["id"] for e in REGISTRY}))

    print()
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    for name, why in SKIPPED.items():
        print(f"skip  {name}: {why}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
