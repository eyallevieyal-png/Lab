"""Regenerate the authoritative simulator summary (repository-root results_summary.json) from the current simulator.

    python -m research.pipeline.make_results_summary
Everything in the file is SIMULATED (no hardware). The metadata block records the simulator version, experiment id, date and configuration.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess

from src.simulator import aggregate, compare_controllers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENT_ID = "EXP-SIM-CTRL-001"
CONFIG = {
    "workloads": ["agentic", "branching"],
    "controllers": ["reactive", "predictive", "wsa", "adaptive", "confidence_gated_adaptive", "oracle"],
    "precisions": ["int4"],
    "num_ops": 24,
    "seeds": [7, 11, 42, 101, 202],
    "calibration_offset": 0.0,
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build() -> dict:
    runs = compare_controllers(**CONFIG)
    summary = [r.to_summary() for r in runs]
    return {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "hypothesis": "H13/H6 controller comparison in the corrected analytical simulator (is prediction worth its cost?)",
            "evidence_class": "simulated (analytical model; not hardware, not measured)",
            "simulator": "src/simulator.py",
            "simulator_commit": _git("log", "-1", "--format=%h", "--", "src/simulator.py", "src/memory.py", "src/workloads.py", "src/config.py") or "uncommitted",
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": CONFIG,
            "runs": len(runs),
            "validity_checks": "python -m research.validation.verify_simulator_fixes (8/8 defects verified fixed)",
            "regenerate": "python -m research.pipeline.make_results_summary",
            "supersedes": "archive/legacy_results/v0_2026-09-17/SUPERSEDED__results_summary.json",
            "units": "all times in milliseconds of modelled time",
        },
        "aggregate": aggregate(summary),
        "summary": summary,
    }


if __name__ == "__main__":
    payload = build()
    with open(os.path.join(ROOT, "results_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
    print("wrote results_summary.json:", payload["metadata"]["runs"], "runs")
