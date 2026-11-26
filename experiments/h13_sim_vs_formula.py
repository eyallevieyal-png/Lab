"""Tests the closed-form break-even accuracy p* = (u + gamma) / (1 + u) (proved in Lean) against the queueing simulation in src/kv_prefetch.py.

For each background utilisation u and predictor cost gamma the simulation finds the accuracy at which predictive prefetching stops stalling more than
reactive fetching (bisection, common random numbers), for two lead times. The formula's assumptions (a wrong prefetch hurts with probability u, a right
one hides the whole transfer) are not built into the simulation, so any gap measures how good those assumptions are.
It also checks that the adaptive policy (act only when p_hat > formula threshold) is never clearly worse than the better of reactive and predictive.

Run: python experiments/h13_sim_vs_formula.py   (writes experiments/results/h13_sim_vs_formula.json)
"""
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kv_prefetch import KVConfig, breakeven_accuracy, closed_form, mean_stall  # noqa: E402

US = [0.0, 0.25, 0.5, 0.75, 0.9]
GAMMAS = [0.0, 0.05]
LEADS = [1.5, 3.0]
SEEDS = [1, 2, 3]


def main():
    out = {"breakeven": [], "adaptive": []}
    print("break-even accuracy: simulation (mean of 3 seeds) vs closed form (u + gamma) / (1 + u)")
    print("(u = background utilisation; 'formula(u_eff)' uses u + 1/period, the link load including this stream's own demand fetches)")
    print(f"{'lead':>5} {'u':>5} {'gamma':>6} {'formula':>8} {'formula(u_eff)':>15} {'simulated':>10} {'gap':>7}")
    for lead in LEADS:
        for g in GAMMAS:
            for u in US:
                vals = [breakeven_accuracy(KVConfig(u_bg=u, gamma=g, lead=lead, seed=s)) for s in SEEDS]
                sim = None if any(v is None for v in vals) else mean(vals)
                f = closed_form(u, g)
                f_eff = closed_form(min(u + 1 / KVConfig().period, 1.0), g)
                out["breakeven"].append({"lead": lead, "u": u, "gamma": g, "formula": f, "formula_u_eff": f_eff, "simulated": sim})
                print(f"{lead:5.1f} {u:5.2f} {g:6.2f} {f:8.3f} {f_eff:15.3f} {('never' if sim is None else f'{sim:10.3f}'):>10} {('' if sim is None else f'{sim - f:+7.3f}')}")
    print("\nadaptive policy vs the better fixed policy (mean stall per epoch in t_x units, 3 seeds; lower is better), gamma = 0.05, lead 1.5")
    print(f"{'u':>5} {'p':>5} {'reactive':>9} {'predictive':>11} {'adaptive':>9} {'best fixed':>11}  adaptive within 0.02 of best?")
    worst = 0.0
    for u in US:
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            r = mean(mean_stall(KVConfig(u_bg=u, gamma=0.05, seed=s), "reactive", p) for s in SEEDS)
            pr = mean(mean_stall(KVConfig(u_bg=u, gamma=0.05, seed=s), "predictive", p) for s in SEEDS)
            ad = mean(mean_stall(KVConfig(u_bg=u, gamma=0.05, seed=s), "adaptive", p) for s in SEEDS)
            best = min(r, pr)
            worst = max(worst, ad - best)
            out["adaptive"].append({"u": u, "p": p, "reactive": r, "predictive": pr, "adaptive": ad})
            print(f"{u:5.2f} {p:5.2f} {r:9.3f} {pr:11.3f} {ad:9.3f} {best:11.3f}  {'yes' if ad - best <= 0.02 else 'NO (+%.3f)' % (ad - best)}")
    out["adaptive_worst_gap"] = worst
    print(f"\nlargest gap of adaptive above the better fixed policy: {worst:+.3f} t_x")
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h13_sim_vs_formula.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
