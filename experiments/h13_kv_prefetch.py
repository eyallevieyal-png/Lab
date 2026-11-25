"""H13 calibrated: when does predicting and prefetching an offloaded KV cache pay off?

Scenario (one decode sequence whose KV cache sits in host memory and is needed again soon):
  reactive   : when the sequence is scheduled the KV cache is fetched over the host link and the step stalls for  T_x = KV_bytes / link_bandwidth.
  predictive : a predictor says "this sequence will be needed", right with probability p, and the cache is prefetched.
               right  -> the stall is hidden (gain B = T_x)
               wrong  -> the wasted transfer occupies the link for T_x and delays a demand fetch with probability u (link utilisation),
                         so the expected loss is L = u * T_x
               cost   -> c = gamma * T_x  (predictor compute, bookkeeping), as a fraction of T_x
  net benefit = p*B - (1-p)*L - c   (the quantity of formal/LabFormal/Orchestration.lean: netBenefit)
  break-even  p* = (u + gamma) / (1 + u)   (proved in Lean: Lab.prefetch_threshold_iff)

Everything about u and gamma is a parameter, not a measurement; T_x comes from sourced specifications. The host link of 128 GB/s is the
figure printed on NVIDIA's H100 page ("PCIe Gen5: 128GB/s"); that is the bidirectional total of a x16 link, so ONE direction is taken as 64 GB/s
(our assumption; both are shown).

Run: python experiments/h13_kv_prefetch.py   (writes experiments/results/h13_kv_prefetch.json)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration.hardware_models import DEVICES, MODELS, kv_bytes_per_token, total_params  # noqa: E402

LINKS_GBPS = {"PCIe5 x16, 64 GB/s one way (assumed)": 64e9, "PCIe5 as printed, 128 GB/s": 128e9}
CONTEXTS = [512, 4096, 32768]
UTILISATIONS = [0.0, 0.25, 0.5, 0.75, 0.9]
GAMMAS = [0.0, 0.05]


def threshold(u: float, gamma: float) -> float:
    return (u + gamma) / (1 + u)


def main():
    hbm = DEVICES["H100 SXM"]["bandwidth_TBps"] * 1e12
    out = {"assumptions": "u and gamma are parameters; link bandwidth from NVIDIA H100 page (128 GB/s printed, 64 GB/s one way assumed)", "rows": [], "thresholds": {}}
    print("T_x = time to fetch one sequence's bf16 KV cache over the host link; step = one batch-1 decode step on H100 SXM (memory-bound, weights + KV from HBM)")
    print(f"{'model':14} {'ctx':>6} {'KV (MiB)':>9} {'step ms':>8} " + " ".join(f"{k.split(',')[1].strip():>14}" for k in LINKS_GBPS) + "   T_x / step")
    for model, m in MODELS.items():
        for ctx in CONTEXTS:
            kv = ctx * kv_bytes_per_token(m, 2)
            step = (total_params(m) * 2 + kv) / hbm
            row = {"model": model, "context": ctx, "kv_bytes": kv, "step_s": step}
            txs = []
            for name, bw in LINKS_GBPS.items():
                row[f"T_x_s[{name}]"] = kv / bw
                txs.append(kv / bw)
            out["rows"].append(row)
            print(f"{model:14} {ctx:6d} {kv / 2**20:9.0f} {step * 1e3:8.2f} " + " ".join(f"{t * 1e3:11.2f} ms" for t in txs) + f"   {txs[0] / step:5.2f} / {txs[1] / step:5.2f}")
    print("\nbreak-even predictor accuracy p* = (u + gamma) / (1 + u), by link utilisation u (rows) and predictor cost gamma (columns):")
    print(f"{'u':>6} " + " ".join(f"gamma={g:<5}" for g in GAMMAS))
    for u in UTILISATIONS:
        print(f"{u:6.2f} " + " ".join(f"{threshold(u, g):10.3f}" for g in GAMMAS))
        out["thresholds"][str(u)] = {str(g): threshold(u, g) for g in GAMMAS}
    print()
    print(f"reading: on an idle link (u = 0) a predictor with cost gamma = 0 pays off at any accuracy; at 90% link utilisation it must be right "
          f"{threshold(0.9, 0.0):.0%} of the time; and prefetching can never pay off once gamma >= 1, i.e. when the predictor costs as much as the stall it hides.")
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h13_kv_prefetch.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
