"""Reproducible benchmark: scenarios x background load x policies x seeds, with 95% confidence intervals and paired comparisons.

    python -X utf8 -m orchestrator.bench            (about 3-5 minutes, CPU only, writes research/results/orchestrator_bench.json and the dashboard)
    python -X utf8 -m orchestrator.bench --quick    (3 seeds, for a smoke check)

Everything is SIMULATED with assumed tier parameters (orchestrator/tiers.py). Same traces, arrivals and background load for all policies of one seed, so the
paired differences below are not sampling noise. `oracle_prefetch` is a clairvoyant upper bound, not a policy anyone can implement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess

from .dashboard import write_dashboard
from .engine import Config, run_one
from .policies import ALL_POLICIES
from .tiers import DEFAULT_TIERS
from .workload import DESCRIPTION, SCENARIOS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = "lru_demand"
BG_LEVELS = {"idle link": {}, "background 50% on the DRAM link": {"DRAM": 0.5}}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045}


def tcrit(df: int) -> float:
    if df < 1:
        return float("nan")
    keys = [k for k in T975 if k <= df]
    return T975[max(keys)] if df < 30 else 1.96


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, tcrit(n - 1) * sd / math.sqrt(n)


METRICS = ("mean_latency_ms", "p99_ms", "throughput_rps", "hbm_block_hit_rate", "bytes_moved_gib", "wasted_prefetch_gib")


def run(seeds):
    raw = []
    for sc in SCENARIOS:
        for bg_name, bg in BG_LEVELS.items():
            for seed in seeds:
                for pol in ALL_POLICIES:
                    r = run_one(Config(scenario=sc, seed=seed, bg_util=dict(bg)), pol)
                    r["bg_name"] = bg_name
                    raw.append(r)
    cells = []
    for sc in SCENARIOS:
        for bg_name in BG_LEVELS:
            cell = {"scenario": sc, "background": bg_name, "policies": {}}
            by = {p: [r for r in raw if r["scenario"] == sc and r["bg_name"] == bg_name and r["policy"] == p] for p in ALL_POLICIES}
            for p, rows in by.items():
                rows.sort(key=lambda r: r["seed"])
                entry = {}
                for m in METRICS:
                    mu, ci = mean_ci([r[m] for r in rows])
                    entry[m] = {"mean": mu, "ci95": ci}
                if p != BASELINE:
                    for m in ("mean_latency_ms", "p99_ms"):
                        d = [a[m] - b[m] for a, b in zip(rows, by[BASELINE])]
                        mu, ci = mean_ci(d)
                        base = sum(b[m] for b in by[BASELINE]) / len(d)
                        entry[f"paired_vs_{BASELINE}_{m}"] = {"diff_mean": mu, "ci95": ci, "pct": 100 * mu / base, "significant": abs(mu) > ci, "lower_is_better": True}
                cell["policies"][p] = entry
            cells.append(cell)
    return raw, cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    seeds = list(range(3 if args.quick else 20))
    raw, cells = run(seeds)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cfg = Config()
    payload = {
        "metadata": {
            "experiment_id": "EXP-ORCH-001", "evidence_class": "simulated (assumed tier parameters; NOT hardware, NOT measured, no CXL hardware involved)",
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "repo_head": head, "seeds": seeds, "quick": args.quick,
            "baseline": BASELINE, "regenerate": "python -X utf8 -m orchestrator.bench",
            "engine": {k: getattr(cfg, k) for k in ("n_sessions", "session_blocks", "block_bytes", "compute_ms", "period_ms", "requests", "warmup")},
            "tiers": [t.__dict__ for t in DEFAULT_TIERS], "scenarios": DESCRIPTION, "backgrounds": BG_LEVELS,
            "statistics": "mean over seeds with t-based 95% CI; paired difference vs baseline on identical traces; 'significant' = CI excludes 0 (no multiple-comparison correction)",
            "predictor": "causal first-order Markov with popularity fallback; accuracy = running share of correct top-1 predictions",
        },
        "cells": cells,
    }
    out = os.path.join(ROOT, "research", "results", "orchestrator_bench" + ("_quick" if args.quick else "") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    write_dashboard(payload, os.path.join(ROOT, "research", "figures", "orchestrator_dashboard" + ("_quick" if args.quick else "") + ".html"))
    for c in cells:
        print(f"\n{c['scenario']}  [{c['background']}]")
        for p, e in c["policies"].items():
            d = e.get(f"paired_vs_{BASELINE}_p99_ms")
            tail = f"  P99 vs baseline {d['pct']:+6.1f}% {'*' if d['significant'] else ' '}" if d else ""
            print(f"  {p:16s} mean {e['mean_latency_ms']['mean']:6.1f}±{e['mean_latency_ms']['ci95']:4.1f}  p99 {e['p99_ms']['mean']:6.1f}±{e['p99_ms']['ci95']:5.1f}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
