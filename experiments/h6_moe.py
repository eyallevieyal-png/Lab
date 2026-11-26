"""H6 for mixture-of-experts models: how expert-union weight reads move the compute/memory balance point (specification-based, no measurement).

Compares each MoE with a dense twin that has the same active parameters per token (all experts always read), for plain decode (spec = 1) and for
speculative verification of `spec` positions per sequence, under uniform and skewed expert popularity.

Run: python experiments/h6_moe.py   (writes experiments/results/h6_moe.json)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration.moe_model import (MOE_MODELS, SOURCES, critical_batch, dense_equivalent, max_batch_in_memory, step, union_skewed,  # noqa: E402
                                   union_uniform)

DEV = "H100 SXM"
SCENARIOS = [("bf16", 16, 16, "bf16"), ("fp8", 8, 8, "fp8")]
CONTEXTS = [512, 4096]
SPECS = [1, 4, 8]
SKEWS = [0.0, 0.5, 1.0]


def fmt(b):
    return "never" if b is None else f"{b:,.0f}"


def main():
    out = {"sources": SOURCES, "device": DEV, "union_fraction": [], "balance": []}
    print(f"expert-union fraction read per layer, uniform routing (tokens in step n):")
    print(f"{'model':14} " + " ".join(f"n={n:<5}" for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)))
    for name, m in MOE_MODELS.items():
        fr = [union_uniform(n, m["experts"], m["topk"]) / m["experts"] for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
        out["union_fraction"].append({"model": name, "n": [1, 2, 4, 8, 16, 32, 64, 128, 256], "fraction": fr})
        print(f"{name:14} " + " ".join(f"{x:7.3f}" for x in fr))
    print(f"\nskew sensitivity (Qwen3-30B-A3B, n=8 tokens): " + ", ".join(f"skew {s}: {union_skewed(8, 128, 8, s) / 128:.3f}" for s in SKEWS) + "  (approximation, assumed)")

    print(f"\n=== {DEV}: balance batch B* (sequences with t_compute = t_memory); spec = positions verified per sequence ===")
    print(f"{'model':14} {'fmt':5} {'ctx':>6} {'spec':>4} {'skew':>5} | {'t_c/t_m@B=1':>11} {'B* MoE':>9} {'B* dense twin':>14} {'factor':>7} {'max B fits':>11}")
    for name, m in MOE_MODELS.items():
        twin = dense_equivalent(m)
        for sname, wb, kvb, f in SCENARIOS:
            for ctx in CONTEXTS:
                mb = max_batch_in_memory(m, DEV, wb, kvb, ctx)
                for spec in SPECS:
                    for skew in SKEWS if spec == 1 else [0.0]:
                        tc, tm = step(m, DEV, wb, kvb, f, 1, ctx, spec, skew)
                        b = critical_batch(m, DEV, wb, kvb, f, ctx, spec, skew)
                        bd = critical_batch(twin, DEV, wb, kvb, f, ctx, spec)
                        factor = None if b is None or bd is None else b / bd
                        out["balance"].append({"model": name, "format": sname, "context": ctx, "spec": spec, "skew": skew, "ratio_b1": tc / tm,
                                               "b_star": b, "b_star_dense_twin": bd, "factor": factor, "max_batch_fits": mb})
                        print(f"{name:14} {sname:5} {ctx:6d} {spec:4d} {skew:5.1f} | {tc / tm:11.4f} {fmt(b):>9} {fmt(bd):>14} {('' if factor is None else f'{factor:6.1f}x'):>7} {('weights do not fit' if mb is None else f'{mb:,}'):>11}")
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h6_moe.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
