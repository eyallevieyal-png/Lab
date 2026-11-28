"""Compare an archived result file with the current simulator, to show whether it came from the defective version.

    python -m research.validation.provenance_check
Prints the table used in archive/legacy_results/v0_2026-09-17/SUPERSEDED.md. Exit code 1 if the archived file looks valid (then the archive label must be reconsidered).
"""
from __future__ import annotations

import collections
import json
import os
import sys

from src.simulator import compare_controllers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD = os.path.join(ROOT, "archive", "legacy_results", "v0_2026-09-17", "SUPERSEDED__results_summary.json")


def mean_by_controller(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["controller"]].append(r["mean_regret"])
    return {k: sum(v) / len(v) for k, v in d.items()}


def main() -> int:
    old = json.load(open(OLD, encoding="utf-8"))["research_report"]
    new = [r.to_summary() for r in compare_controllers(workloads=["agentic", "branching"], precisions=["int4"], num_ops=24, seeds=[7, 11])]
    mo, mn = mean_by_controller(old), mean_by_controller(new)
    print(f"{'controller':28s} {'archived v0':>12s} {'current':>10s}")
    for c in sorted(mo):
        print(f"{c:28s} {mo[c]:12.4f} {mn.get(c, float('nan')):10.4f}")
    oracle_positive = mo["Oracle"] > 1e-9
    oracle_beats_reactive = mo["Oracle"] < mo["Reactive"]
    seed_effect = len({r["p99"] for r in old if r["controller"] == "Reactive" and r["workload"] == "agentic"}) > 1
    print(f"\nv0 oracle regret > 0: {oracle_positive} (a real oracle has 0)   v0 oracle regret < reactive: {oracle_beats_reactive}   v0 seed changes Reactive p99: {seed_effect}")
    print(f"current oracle regret: {mn['Oracle']:.4f}")
    looks_valid = (not oracle_positive) and oracle_beats_reactive and seed_effect
    return 1 if looks_valid else 0


if __name__ == "__main__":
    sys.exit(main())
