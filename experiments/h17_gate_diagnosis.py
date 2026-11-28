"""H17 diagnosis: why is the congestion-aware H13 gate worse than a static accuracy threshold on a predictable workload under background load?

    python -X utf8 experiments/h17_gate_diagnosis.py        (about a minute, CPU only; writes experiments/results/h17_gate_diagnosis.json)

Variants of the gate isolate one suspect each. Seeds 100-119 are used for this diagnosis, separate from the benchmark seeds 0-19, so nothing is tuned on the
evaluation seeds. SIMULATED (orchestrator prototype, assumed tier parameters). Paired differences on identical traces.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator.bench import mean_ci  # noqa: E402
from orchestrator.engine import Config, Engine  # noqa: E402
from orchestrator.policies import AdaptiveGate, AlwaysPrefetch, LRUDemand, StaticThreshold  # noqa: E402

SEEDS = list(range(100, 120))
BG = {"DRAM": 0.5}


class GateTrueU(AdaptiveGate):
    """Uses the TRUE background utilisation instead of the measured one: separates estimation error from model error."""
    name = "gate_true_u"

    def prefetch(self, ctx, missing, est):
        ctx.util = {t: BG.get(t, 0.0) for t in ctx.util}
        return super().prefetch(ctx, missing, est)


class GateU0(AdaptiveGate):
    name = "gate_u_zero"

    def prefetch(self, ctx, missing, est):
        ctx.util = {t: 0.0 for t in ctx.util}
        return super().prefetch(ctx, missing, est)


class GateNoQueue(AdaptiveGate):
    name = "gate_no_queue_check"
    max_queue_tx = float("inf")


class GateNoMargin(AdaptiveGate):
    name = "gate_no_margin"
    margin = 0.0


VARIANTS = [LRUDemand, AlwaysPrefetch, StaticThreshold, AdaptiveGate, GateTrueU, GateU0, GateNoQueue, GateNoMargin]


def main() -> None:
    rows = {v.name: [] for v in VARIANTS}
    for seed in SEEDS:
        for v in VARIANTS:
            rows[v.name].append(Engine(Config(scenario="agentic_loop", seed=seed, bg_util=dict(BG)), v()).run())
    out = {"experiment": "H17 gate diagnosis", "evidence_class": "simulated", "scenario": "agentic_loop", "background": BG, "seeds": SEEDS, "variants": {}}
    ref = rows["always_prefetch"]
    for name, rs in rows.items():
        p99, ci = mean_ci([r["p99_ms"] for r in rs])
        d, dci = mean_ci([a["p99_ms"] - b["p99_ms"] for a, b in zip(rs, ref)])
        out["variants"][name] = {"p99_ms": p99, "p99_ci95": ci, "mean_latency_ms": mean_ci([r["mean_latency_ms"] for r in rs])[0],
                                 "prefetches_issued": mean_ci([r["prefetches_issued"] for r in rs])[0], "wasted_prefetch_gib": mean_ci([r["wasted_prefetch_gib"] for r in rs])[0],
                                 "p99_minus_always_prefetch_ms": d, "p99_minus_always_ci95": dci}
        print(f"{name:22s} P99 {p99:6.1f}±{ci:4.1f}  vs always_prefetch {d:+6.1f}±{dci:4.1f} ms  prefetches {out['variants'][name]['prefetches_issued']:7.0f}  wasted {out['variants'][name]['wasted_prefetch_gib']:5.1f} GiB")
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "results", "h17_gate_diagnosis.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
