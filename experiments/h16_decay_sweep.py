"""H16: is the phase_shift regression of reuse-score placement caused by a score that decays too slowly after a working-set change?

    python -X utf8 experiments/h16_decay_sweep.py              (tuning seeds 100-119, about 3 minutes)
    python -X utf8 experiments/h16_decay_sweep.py --confirm    (held-out seeds 0-19)
Writes experiments/results/h16_decay_sweep[_confirm].json. SIMULATED; assumed tier parameters; paired differences vs the LRU baseline.

Only placement is tested (no prefetching: both policies below never prefetch), so the effect of the eviction and demotion rules is isolated.
`placement` uses the reuse-score eviction and demotion of `Adaptive`; the only thing that varies is the per-request score decay (Config.decay; 0.97 in the benchmark).
Prediction (written before running): a faster decay shrinks the phase_shift regression and costs something on zipf_iid, where long memory matters. If both hold, the
mechanism is confirmed; if the regression does not shrink, the stale-score hypothesis is wrong.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from orchestrator.bench import mean_ci  # noqa: E402
from orchestrator.engine import Config, Engine  # noqa: E402
from orchestrator.policies import Adaptive, LRUDemand, Policy  # noqa: E402

CONFIRM = "--confirm" in sys.argv
SEEDS = list(range(0, 20)) if CONFIRM else list(range(100, 120))
DECAYS = [0.7, 0.85, 0.93, 0.97, 0.99]
SCENARIOS = ["zipf_iid", "phase_shift", "uniform", "agentic_loop"]


class Placement(Adaptive):
    name = "placement_only"

    def prefetch(self, ctx, missing_by_tier, est):
        return None


class EvictOnly(Placement):
    """Reuse-score eviction from HBM; plain LRU waterfall below."""
    name = "evict_only"

    def lower_victim(self, tier, lru_order, ctx):
        return lru_order[0]

    def demote_tier(self, block, ctx, free):
        return Policy.demote_tier(self, block, ctx, free)


class DemoteOnly(Placement):
    """LRU eviction from HBM; reuse-score demotion tier and lower-tier victims."""
    name = "demote_only"

    def hbm_victim(self, lru_order, ctx):
        return lru_order[0]


def ablation(out: dict) -> None:
    print("\nAblation: which part of the placement rule causes the regression? (P99 change vs LRU)")
    out["ablation"] = []
    for sc in ("phase_shift", "zipf_iid"):
        base = [Engine(Config(scenario=sc, seed=s), LRUDemand()).run() for s in SEEDS]
        for dcy in (0.85, 0.97):
            for v in (Placement, EvictOnly, DemoteOnly):
                rs = [Engine(Config(scenario=sc, seed=s, decay=dcy), v()).run() for s in SEEDS]
                d, ci = mean_ci([a["p99_ms"] - b["p99_ms"] for a, b in zip(rs, base)])
                print(f"  {sc:12s} decay {dcy}  {v.name:15s} {d:+6.2f} ± {ci:4.2f} ms")
                out["ablation"].append({"scenario": sc, "decay": dcy, "variant": v.name, "p99_vs_lru_ms": d, "p99_ci95": ci})


def main() -> None:
    out = {"experiment": "H16 decay sweep", "evidence_class": "simulated", "seeds": SEEDS, "confirmation_run": CONFIRM, "cells": []}
    for sc in SCENARIOS:
        base = [Engine(Config(scenario=sc, seed=s), LRUDemand()).run() for s in SEEDS]
        print(f"\n{sc}   (LRU baseline P99 {mean_ci([r['p99_ms'] for r in base])[0]:.1f} ms)   P99 change vs LRU, negative = better")
        cell = {"scenario": sc, "decays": {}}
        for dcy in DECAYS:
            rs = [Engine(Config(scenario=sc, seed=s, decay=dcy), Placement()).run() for s in SEEDS]
            d, ci = mean_ci([a["p99_ms"] - b["p99_ms"] for a, b in zip(rs, base)])
            dm, cim = mean_ci([a["mean_latency_ms"] - b["mean_latency_ms"] for a, b in zip(rs, base)])
            hit = mean_ci([r["hbm_block_hit_rate"] for r in rs])[0]
            cell["decays"][str(dcy)] = {"p99_vs_lru_ms": d, "p99_ci95": ci, "mean_vs_lru_ms": dm, "mean_ci95": cim, "hbm_hit_rate": hit}
            print(f"  decay {dcy:4.2f}: P99 {d:+6.2f} ± {ci:4.2f} ms   mean {dm:+5.2f} ± {cim:4.2f} ms   HBM hit {hit:.3f}")
        out["cells"].append(cell)
    ablation(out)
    with open(os.path.join(ROOT, "experiments", "results", "h16_decay_sweep_confirm.json" if CONFIRM else "h16_decay_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
