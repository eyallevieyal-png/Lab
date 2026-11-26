"""Load-aware (online) prefetch controller under real bursty background traffic (BurstGPT arrival pattern) and under Poisson traffic.

The online controller estimates the current background load from the last 30 transfer times of observed arrivals (causal) and prefetches only when the predictor's
accuracy exceeds the congestion-aware M/D/1 break-even for that load (src/kv_prefetch.py: analytic_breakeven); it never prefetches when reactive fetching alone would
saturate the link. It is compared with always-reactive and always-prefetch (mean stall per epoch in units of one KV transfer; lower is better).
The trace is untrusted data parsed as numbers only. Run: python -I experiments/h13_online_controller.py <BurstGPT csv>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.h13_trace_background import WINDOW, load_timestamps  # noqa: E402
from src.kv_prefetch import KVConfig, mean_stall  # noqa: E402
import random  # noqa: E402

PERIOD, LEAD, GAMMA, EPOCHS = 12.0, 1.5, 0.05, 3000
US = [0.5, 0.65]
PS = [0.3, 0.6, 0.9]


def main(path):
    ts = load_timestamps(path)
    rng = random.Random("burstgpt-jitter")
    jit = sorted(t + rng.random() for t in ts)
    gaps = [b - a for a, b in zip(jit, jit[1:])]
    sources = {"Poisson": None}
    for name, f in (("trace win 1 (smooth)", 0.1), ("trace win 3 (bursty)", 0.9), ("trace win 2 (very bursty)", 0.5)):
        st = int(len(gaps) * f)
        sources[name] = tuple(gaps[st:st + WINDOW])
    out = []
    print(f"mean stall per epoch (transfer times), period {PERIOD:g}, lead {LEAD:g}, gamma {GAMMA:g}")
    print(f"{'traffic':26} {'u':>5} {'p':>4} | {'reactive':>9} {'always':>9} {'online':>9} | online vs best fixed")
    for name, g in sources.items():
        for u in US:
            for p in PS:
                cfg = KVConfig(u_bg=u, gamma=GAMMA, lead=LEAD, period=PERIOD, epochs=EPOCHS, bg_gaps=g)
                r, a, o = (mean_stall(cfg, pol, p) for pol in ("reactive", "predictive", "online"))
                best = min(r, a)
                out.append({"traffic": name, "u": u, "p": p, "reactive": r, "always": a, "online": o})
                verdict = "better" if o < best - 0.02 * best - 0.01 else ("same" if o <= best * 1.05 + 0.02 else f"WORSE (+{o - best:.2f})")
                print(f"{name:26} {u:5.2f} {p:4.1f} | {r:9.2f} {a:9.2f} {o:9.2f} | {verdict}")
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h13_online_controller.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
