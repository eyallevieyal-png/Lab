"""H17 diagnosis, step 2: which assumption of the congestion-aware H13 gate makes it trail a static accuracy threshold?

    python -X utf8 experiments/h17_gate_diagnosis2.py     (about 4 minutes, CPU only; writes experiments/results/h17_gate_diagnosis2.json)

Step 1 (h17_gate_diagnosis.py) showed the gap is not the load estimate. Instrumenting the gate showed that its threshold jumps between about 0.22 and 0.64 depending on
which tier the missing blocks sit in (the `lead` expressed in transfer times changes), and that under 50% background it skips about 24% of the candidate prefetches
(static threshold: about 3%). Variants, each changing ONE thing:

  gate                  the H13 congestion-aware gate as in the benchmark
  gate_fixed_lead       lead and period from a fixed reference transfer time (removes the tier-dependent jump)
  gate_cond_conf        compares the model's CONDITIONAL confidence for this prediction (not the global running accuracy) with the threshold
  gate_acc_slow/fast    running-accuracy window 0.01 / 0.2 instead of 0.05
  static_cond_conf      static rule: conditional confidence > 0.5 (a competitor made from the same ingredient as gate_cond_conf)
  static_threshold      static rule: global accuracy > 0.5 (as in the benchmark)

Tuning-free: nothing is fitted; seeds 100-119 only (not the benchmark seeds 0-19). SIMULATED; assumed tier parameters. Paired differences vs always_prefetch.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from orchestrator.bench import mean_ci  # noqa: E402
from orchestrator.engine import Config, Engine  # noqa: E402
from orchestrator.policies import AdaptiveGate, AlwaysPrefetch, LRUDemand, StaticThreshold, _breakeven  # noqa: E402
from orchestrator.telemetry import MarkovPredictor  # noqa: E402

SEEDS = list(range(100, 120))
CELLS = [("agentic_loop", {}), ("agentic_loop", {"DRAM": 0.5}), ("uniform", {"DRAM": 0.5}), ("phase_shift", {"DRAM": 0.5})]
REF_TX_MS = 8 * (0.001 + 32 * 1024 * 1024 / (64 * 1e9) * 1000)     # one session of 8 blocks over the DRAM link


class GateFixedLead(AdaptiveGate):
    name = "gate_fixed_lead"

    def prefetch(self, ctx, missing, est):
        c = self._candidate(ctx)
        need = missing.get(c, {}) if c is not None else {}
        if not need:
            return None
        tier = max(need, key=lambda t: est.stream_ms(t, need[t]))
        t_x = est.stream_ms(tier, need[tier])
        if ctx.queue_ms[tier] > self.max_queue_tx * t_x:
            return None
        u = round(min(0.95, ctx.util[tier]) * 20) / 20
        lead = period = round(ctx.mean_gap_ms / REF_TX_MS * 4) / 4
        thr = _breakeven(u, self.gamma, lead, period)
        return c if thr is not None and ctx.pred_accuracy > thr + self.margin else None


class GateCondConf(AdaptiveGate):
    name = "gate_cond_conf"

    def prefetch(self, ctx, missing, est):
        ctx.pred_accuracy = ctx.pred_p_model
        return super().prefetch(ctx, missing, est)


class StaticCondConf(StaticThreshold):
    name = "static_cond_conf"

    def prefetch(self, ctx, missing, est):
        c = self._candidate(ctx)
        return c if c is not None and ctx.pred_p_model > 0.5 else None


class GateAccSlow(AdaptiveGate):
    name = "gate_acc_slow"
    ewma = 0.01


class GateAccFast(AdaptiveGate):
    name = "gate_acc_fast"
    ewma = 0.2


class FixedNoQueue(GateFixedLead):
    name = "fixed_lead+no_queue_check"
    max_queue_tx = float("inf")


class FixedNoQueueU0(FixedNoQueue):
    name = "fixed_lead+no_queue+u_ignored"

    def prefetch(self, ctx, missing, est):
        ctx.util = {t: 0.0 for t in ctx.util}
        return super().prefetch(ctx, missing, est)


class FixedNoQueueTrueU(FixedNoQueue):
    name = "fixed_lead+no_queue+true_u"

    def prefetch(self, ctx, missing, est):
        ctx.util = {t: (0.5 if t == "DRAM" else 0.0) for t in ctx.util}
        return super().prefetch(ctx, missing, est)


COMBINED = [GateFixedLead, FixedNoQueue, FixedNoQueueU0, FixedNoQueueTrueU]

VARIANTS = [LRUDemand, AlwaysPrefetch, StaticThreshold, AdaptiveGate, GateFixedLead, GateCondConf, StaticCondConf, GateAccSlow, GateAccFast]


def run(cell, v, seed):
    sc, bg = cell
    e = Engine(Config(scenario=sc, seed=seed, bg_util=dict(bg)), v())
    if getattr(v, "ewma", None):
        e.pred = MarkovPredictor(ewma=v.ewma)
    return e.run()


def main() -> None:
    out = {"experiment": "H17 gate diagnosis 2", "evidence_class": "simulated", "seeds": SEEDS, "cells": []}
    for cell in CELLS:
        rows = {v.name: [run(cell, v, s) for s in SEEDS] for v in VARIANTS}
        ref = rows["always_prefetch"]
        entry = {"scenario": cell[0], "background": cell[1], "variants": {}}
        print(f"\n{cell[0]} bg={cell[1]}")
        for name, rs in rows.items():
            p99, ci = mean_ci([r["p99_ms"] for r in rs])
            d, dci = mean_ci([a["p99_ms"] - b["p99_ms"] for a, b in zip(rs, ref)])
            entry["variants"][name] = {"p99_ms": p99, "p99_ci95": ci, "vs_always_ms": d, "vs_always_ci95": dci,
                                       "prefetches": mean_ci([r["prefetches_issued"] for r in rs])[0], "wasted_gib": mean_ci([r["wasted_prefetch_gib"] for r in rs])[0]}
            print(f"  {name:18s} P99 {p99:6.1f}±{ci:4.1f}  vs always {d:+6.1f}±{dci:4.1f}  prefetches {entry['variants'][name]['prefetches']:6.0f}  wasted {entry['variants'][name]['wasted_gib']:6.1f} GiB")
        out["cells"].append(entry)
    cell = ("agentic_loop", {"DRAM": 0.5})
    ref = [run(cell, AlwaysPrefetch, sd) for sd in SEEDS]
    out["combined_fixes_agentic_loop_background_0.5"] = {}
    print("\ncombined fixes, agentic_loop, background 0.5 (P99 minus always_prefetch)")
    for v in COMBINED:
        rs = [run(cell, v, sd) for sd in SEEDS]
        d, ci = mean_ci([a["p99_ms"] - b["p99_ms"] for a, b in zip(rs, ref)])
        out["combined_fixes_agentic_loop_background_0.5"][v.name] = {"vs_always_ms": d, "ci95": ci}
        print(f"  {v.name:32s} {d:+6.1f} ± {ci:4.1f} ms")
    with open(os.path.join(ROOT, "experiments", "results", "h17_gate_diagnosis2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
