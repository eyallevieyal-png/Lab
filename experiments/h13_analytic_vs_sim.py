"""Congestion-aware break-even accuracy (M/D/1 analytic model) against the queueing simulation, and against the plain formula (u + gamma) / (1 + u).

Grid: background utilisation u, epoch period (own demand-fetch load 1/period), predictor cost gamma, lead time. The simulation is the reference;
we report mean and maximum absolute error of both formulas where the simulation finds a break-even, and whether they agree on "no accuracy pays off".

Run: python experiments/h13_analytic_vs_sim.py   (writes experiments/results/h13_analytic_vs_sim.json)
"""
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kv_prefetch import KVConfig, analytic_breakeven, breakeven_accuracy, closed_form, stability_floor  # noqa: E402

US = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 0.9]
PERIODS = [4.0, 6.0, 12.0]
GAMMAS = [0.0, 0.05]
LEADS = [1.5, 3.0]
SEEDS = [1, 2]


def fmt(x):
    return "never" if x is None else f"{x:.3f}"


def main():
    rows = []
    print(f"{'period':>6} {'lead':>5} {'gamma':>6} {'u':>5} | {'simulated':>9} {'M/D/1':>7} {'plain':>7} | {'stab.floor':>10}")
    for period in PERIODS:
        for lead in LEADS:
            for g in GAMMAS:
                for u in US:
                    vals = [breakeven_accuracy(KVConfig(u_bg=u, gamma=g, lead=lead, period=period, seed=s, epochs=4000)) for s in SEEDS]
                    sim = None if any(v is None for v in vals) else mean(vals)
                    ana = analytic_breakeven(u, g, lead, period)
                    plain = closed_form(u, g)
                    floor = stability_floor(u, period)
                    rows.append({"period": period, "lead": lead, "gamma": g, "u": u, "simulated": sim, "md1": ana, "plain": plain, "stability_floor": floor})
                    print(f"{period:6.1f} {lead:5.1f} {g:6.2f} {u:5.2f} | {fmt(sim):>9} {fmt(ana):>7} {plain:7.3f} | {floor:10.3f}")
    # cells where reactive fetching alone saturates the link (u + 1/period >= 1) are unstable for every policy: reported separately, excluded from the error
    saturated = [r for r in rows if r["u"] + 1 / r["period"] >= 1]
    stable = [r for r in rows if r["u"] + 1 / r["period"] < 1]
    both = [r for r in stable if r["simulated"] is not None and r["md1"] is not None]
    e_md1 = [abs(r["simulated"] - r["md1"]) for r in both]
    e_plain = [abs(r["simulated"] - r["plain"]) for r in both]
    summary = {"cells_total": len(rows), "cells_reactive_saturated": len(saturated), "cells_compared": len(both),
               "mean_abs_error_md1": mean(e_md1), "max_abs_error_md1": max(e_md1), "mean_abs_error_plain": mean(e_plain), "max_abs_error_plain": max(e_plain),
               "sim_never_in_stable_cells": sum(1 for r in stable if r["simulated"] is None), "md1_never_in_stable_cells": sum(1 for r in stable if r["md1"] is None)}
    by_u = {}
    for r in both:
        by_u.setdefault(round(r["u"] + 1 / r["period"], 2), []).append(abs(r["simulated"] - r["md1"]))
    print()
    print(f"{len(saturated)} of {len(rows)} cells have a reactive policy that already saturates the link (excluded). Over the {len(both)} compared cells:")
    print("  mean |error| M/D/1 = %.3f (max %.3f);  plain formula = %.3f (max %.3f)" % (summary["mean_abs_error_md1"], summary["max_abs_error_md1"], summary["mean_abs_error_plain"], summary["max_abs_error_plain"]))
    hi = [r for r in both if r["u"] + 1 / r["period"] >= 0.7]
    lo = [r for r in both if r["u"] + 1 / r["period"] < 0.7]
    for name, grp in (("reactive load < 0.7", lo), ("reactive load 0.7-1", hi)):
        if grp:
            print(f"  {name}: {len(grp)} cells, mean |error| M/D/1 {mean(abs(r['simulated'] - r['md1']) for r in grp):.3f}, plain {mean(abs(r['simulated'] - r['plain']) for r in grp):.3f}")
            summary[f"mean_error_md1_{name}"] = mean(abs(r['simulated'] - r['md1']) for r in grp)
            summary[f"mean_error_plain_{name}"] = mean(abs(r['simulated'] - r['plain']) for r in grp)
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h13_analytic_vs_sim.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
