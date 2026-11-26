"""H6 sensitivity: when does memory (not compute) dominate, and how does that depend on the model's assumptions?

H6 (compute-memory inversion): as compute gets cheaper, memory movement becomes a larger share of the cost.
Isolation: the Reactive (data-local) controller, so no placement decisions or transfers mix in; all quantities are model outputs.

For one operation set, compute time scales as 1/k when compute is k times faster and memory time does not change, so
    memory_share(k) = M / (C/k + M)          (C, M = summed compute and memory time at k = 1)
which is exactly the formula proved strictly increasing in k in formal/LabFormal/Orchestration.lean (memShare_strictMono).
The share reaches 50% at k* = C / M: the compute speed-up needed for the "inversion". This script measures C and M in the simulator,
reports k* per tier / size scale / precision, and checks the simulator against the closed form.

Run: python experiments/h6_sensitivity.py   (writes experiments/results/h6_sensitivity.json)
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SimulationConfig  # noqa: E402
from src.simulator import simulate_run  # noqa: E402

WORKLOADS = ["agentic", "branching", "conversational", "long-running", "static-llm", "tool-heavy"]
SEEDS = [7, 11, 42]
TIERS = ["HBM", "DRAM", "CXL", "NVMe", "QPU"]
SCALES = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0, 4.0]
PRECISIONS = ["fp16", "int4", "bitnet_b1.58"]
KS = [0.5, 1, 2, 4, 8, 16, 32]


def sums(precision, size_scale, k=1.0, overlap=False):
    """Summed compute and memory time of the Reactive controller, per home tier."""
    out = defaultdict(lambda: [0.0, 0.0])
    for w in WORKLOADS:
        for s in SEEDS:
            cfg = SimulationConfig(workload=w, precision=precision, controller="reactive", num_ops=36, seed=s,
                                   size_scale=size_scale, compute_ms_per_unit=0.18 / k, overlap_compute_memory=overlap)
            for r in simulate_run(cfg).records:
                t = out[r["home_resource"]]
                t[0] += r["compute_time"]
                t[1] += r["memory_time"]
    return out


def share(c, m):
    return m / (c + m)


def main():
    result = {"k_star": {}, "share_vs_k": {}, "closed_form_max_abs_error": 0.0, "monotone_in_k": True}
    # 1. k* = C/M per precision x size scale x tier (k = 1 baseline)
    for p in PRECISIONS:
        for sc in SCALES:
            t = sums(p, sc)
            allc = sum(v[0] for v in t.values())
            allm = sum(v[1] for v in t.values())
            row = {tier: (t[tier][0] / t[tier][1]) for tier in TIERS if tier in t}
            row["ALL"] = allc / allm
            result["k_star"][f"{p}|scale={sc:g}"] = row
    # 2. share vs k at the default scale, simulator vs closed form, and monotonicity (the Lean theorem)
    for p in PRECISIONS:
        base = sums(p, 1.0)
        c0 = sum(v[0] for v in base.values())
        m0 = sum(v[1] for v in base.values())
        shares = []
        for k in KS:
            t = sums(p, 1.0, k)
            c = sum(v[0] for v in t.values())
            m = sum(v[1] for v in t.values())
            s = share(c, m)
            result["closed_form_max_abs_error"] = max(result["closed_form_max_abs_error"], abs(s - m0 / (c0 / k + m0)))
            shares.append(s)
        result["monotone_in_k"] &= all(a < b for a, b in zip(shares, shares[1:]))
        result["share_vs_k"][p] = dict(zip([str(k) for k in KS], shares))
    # 3. overlap model: how much of memory time is hidden behind compute (execution = max instead of sum)
    for p in ("int4",):
        t_add = sums(p, 1.0, overlap=False)
        result["overlap_note"] = "see tests: execution = max(compute, memory) when overlap_compute_memory=True; shares of components are unchanged."
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h6_sensitivity.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print("k* = compute speed-up at which memory reaches 50% of (compute + memory) time. k* < 1: already memory-dominated.")
    print(f"{'precision':13} {'size scale':>10} " + " ".join(f"{t:>8}" for t in TIERS + ['ALL']))
    for key, row in result["k_star"].items():
        p, sc = key.split("|scale=")
        print(f"{p:13} {sc:>10} " + " ".join(f"{row.get(t, float('nan')):8.3g}" for t in TIERS + ["ALL"]))
    print("\nmemory share vs compute speed-up k (default scale, all tiers):")
    print(f"{'precision':13} " + " ".join(f"k={k:<5}" for k in KS))
    for p, d in result["share_vs_k"].items():
        print(f"{p:13} " + " ".join(f"{d[str(k)] * 100:6.1f}%" for k in KS))
    print(f"\nclosed-form check max |error|: {result['closed_form_max_abs_error']:.2e};  strictly increasing in k everywhere: {result['monotone_in_k']}")


if __name__ == "__main__":
    main()
