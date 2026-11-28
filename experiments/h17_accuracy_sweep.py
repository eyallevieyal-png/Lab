"""H17, the regime that matters: medium predictor accuracy under heavy background load.

    python -X utf8 experiments/h17_accuracy_sweep.py      (about 5 minutes, CPU only; writes experiments/results/h17_accuracy_sweep.json)

Every earlier cell had either high accuracy (0.84) or almost none (0.04), where a static threshold of 0.5 is trivially right. A load-aware gate can only earn its
complexity where the best fixed threshold CHANGES with the load. Question: is there a cell in which the congestion-aware gate beats every static threshold?

Predictor accuracy is controlled by the probability of following the fixed successor (scenario `agentic_loop@p`); the predictor is the same online Markov model.
Background load u on the DRAM link in {0.5, 0.75}. Static thresholds 0.2 / 0.35 / 0.5 / 0.65. `gate` is the benchmark gate; `gate_fixed` has the tier-dependent lead and the
queue check removed (the two defects found in step 2). Seeds 100-119 (not the benchmark seeds). SIMULATED, assumed tier parameters, paired differences vs the LRU baseline.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.h17_gate_diagnosis2 import GateFixedLead  # noqa: E402
from orchestrator.bench import mean_ci  # noqa: E402
from orchestrator.engine import Config, Engine  # noqa: E402
from orchestrator.policies import AdaptiveGate, AlwaysPrefetch, LRUDemand, _PrefetchBase  # noqa: E402

CONFIRM = "--confirm" in sys.argv          # held-out seeds 0-19: the variants were designed while looking at seeds 100-119
SEEDS = list(range(0, 20)) if CONFIRM else list(range(100, 120))
FOLLOW = [0.3, 0.5, 0.7]
BGS = [0.5, 0.75]


def static(th):
    class S(_PrefetchBase):
        name = f"static_{th}"

        def prefetch(self, ctx, missing, est):
            c = self._candidate(ctx)
            return c if c is not None and ctx.pred_accuracy > th else None
    return S


class GateFixedNQ(GateFixedLead):
    name = "gate_fixed"
    max_queue_tx = float("inf")


VARIANTS = [LRUDemand, AlwaysPrefetch, static(0.2), static(0.35), static(0.5), static(0.65), AdaptiveGate, GateFixedNQ]


def main() -> None:
    out = {"experiment": "H17 accuracy sweep", "evidence_class": "simulated", "seeds": SEEDS, "confirmation_run": CONFIRM, "cells": []}
    for f in FOLLOW:
        for bg in BGS:
            sc = f"agentic_loop@{f}"
            rows = {v.name: [Engine(Config(scenario=sc, seed=s, bg_util={"DRAM": bg}), v()).run() for s in SEEDS] for v in VARIANTS}
            base = rows["lru_demand"]
            acc = mean_ci([r["predictor_accuracy"] for r in rows["always_prefetch"]])[0]
            cell = {"follow_prob": f, "background": bg, "predictor_accuracy": acc, "variants": {}}
            print(f"\nfollow={f} (measured predictor accuracy {acc:.2f}), background u={bg}   [P99 change vs LRU baseline, negative = better]")
            best_static = min((n for n in rows if n.startswith("static_")), key=lambda n: sum(r["p99_ms"] for r in rows[n]))
            for name, rs in rows.items():
                d, ci = mean_ci([a["p99_ms"] - b["p99_ms"] for a, b in zip(rs, base)])
                p99 = mean_ci([r["p99_ms"] for r in rs])[0]
                cell["variants"][name] = {"p99_ms": p99, "vs_lru_ms": d, "vs_lru_ci95": ci, "prefetches": mean_ci([r["prefetches_issued"] for r in rs])[0]}
                mark = "  <- best static" if name == best_static else ""
                print(f"  {name:18s} P99 {p99:7.1f}   vs LRU {d:+7.1f}±{ci:5.1f}  prefetches {cell['variants'][name]['prefetches']:6.0f}{mark}")
            cell["best_static"] = best_static
            d, ci = mean_ci([a["p99_ms"] - b["p99_ms"] for a, b in zip(rows["gate_fixed"], rows[best_static])])
            cell["gate_fixed_minus_best_static_in_hindsight"] = {"diff_ms": d, "ci95": ci, "significant": abs(d) > ci}
            print(f"  gate_fixed minus best static (chosen in hindsight, paired): {d:+.1f} ± {ci:.1f} ms  {'SIGNIFICANT' if abs(d) > ci else 'not significant'}")
            out["cells"].append(cell)
    with open(os.path.join(ROOT, "experiments", "results", "h17_accuracy_sweep_confirm.json" if CONFIRM else "h17_accuracy_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
