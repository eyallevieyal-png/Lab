"""H13 threshold under REAL bursty background traffic: the BurstGPT trace (public Azure OpenAI request log, https://github.com/HPMLL/BurstGPT) drives the
background transfers instead of a Poisson process, rescaled so that its average load equals the utilisation u. The trace is untrusted data: it is only
parsed as numbers (csv module), never executed.

The trace has no KV-reuse or session information, so only the ARRIVAL pattern of requests is used (as the pattern of competing link traffic); this does not say
anything about which KV caches are reused. Timestamps have 1 s resolution, so each is jittered uniformly within its second (seeded) before taking gaps.

Run: python -I experiments/h13_trace_background.py <path to BurstGPT_without_fails_1.csv>   (writes experiments/results/h13_trace_background.json)
"""
import csv
import json
import os
import random
import sys
from statistics import mean, pstdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kv_prefetch import KVConfig, analytic_breakeven, breakeven_accuracy  # noqa: E402

WINDOW = 30000
PERIOD = 12.0
LEAD = 1.5
GAMMA = 0.05
US = [0.25, 0.5, 0.65, 0.75]


def load_timestamps(path):
    ts = []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            ts.append(float(row[0]))
    return ts


def main(path):
    ts = load_timestamps(path)
    rng = random.Random("burstgpt-jitter")
    jit = sorted(t + rng.random() for t in ts)
    gaps_all = [b - a for a, b in zip(jit, jit[1:])]
    m = mean(gaps_all)
    cv2 = (pstdev(gaps_all) / m) ** 2
    hours = {}
    for t in jit:
        hours[int(t // 3600)] = hours.get(int(t // 3600), 0) + 1
    hr = list(hours.values())
    print(f"trace: {len(ts):,} requests over {(jit[-1] - jit[0]) / 86400:.1f} days, mean gap {m:.3f} s, inter-arrival CV^2 = {cv2:.2f} (Poisson = 1), hourly-count CV = {pstdev(hr) / mean(hr):.2f}")
    out = {"requests": len(ts), "days": (jit[-1] - jit[0]) / 86400, "interarrival_cv2": cv2, "hourly_count_cv": pstdev(hr) / mean(hr), "rows": []}
    starts = [int(len(gaps_all) * f) for f in (0.1, 0.5, 0.9)]
    print(f"\nbreak-even accuracy, period {PERIOD:g}, lead {LEAD:g}, gamma {GAMMA:g}; trace windows of {WINDOW:,} requests starting at 10%, 50%, 90% of the trace")
    print(f"{'u':>5} | {'Poisson sim':>11} {'M/D/1':>7} | {'trace win 1':>11} {'win 2':>7} {'win 3':>7}")
    for u in US:
        pois = breakeven_accuracy(KVConfig(u_bg=u, gamma=GAMMA, lead=LEAD, period=PERIOD, epochs=3000))
        ana = analytic_breakeven(u, GAMMA, LEAD, PERIOD)
        trs = []
        for st in starts:
            g = tuple(gaps_all[st:st + WINDOW])
            trs.append(breakeven_accuracy(KVConfig(u_bg=u, gamma=GAMMA, lead=LEAD, period=PERIOD, epochs=3000, bg_gaps=g)))
        out["rows"].append({"u": u, "poisson": pois, "md1": ana, "trace_windows": trs})
        f = lambda x: "never" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"{u:5.2f} | {f(pois):>11} {f(ana):>7} | " + " ".join(f"{f(t):>7}" for t in trs))
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h13_trace_background.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
