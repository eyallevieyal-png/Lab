"""Aggregate several runs of experiments/measure_gpu.py on the SAME GPU into means with 95% confidence intervals and a stability check.

    python -X utf8 -m experiments.aggregate_gpu_repeats run1.json run2.json run3.json [--allow-quick] [--out aggregated.json]

One run is an anecdote. This script is what turns "approximately 276 GB/s" into "mean +- CI over N runs, coefficient of variation c". It refuses to mix different
GPUs, torch/CUDA versions or quick (smoke-level) runs unless told to, and it reports the spread instead of hiding it.
Pure Python: no GPU, no torch needed to aggregate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 29: 2.045}


def _t(df: int) -> float:
    if df < 1:
        return float("nan")
    return 1.96 if df >= 30 else T975[max(k for k in T975 if k <= df)]


def stat(xs: List[float]) -> Dict[str, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    ci = _t(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "ci95": ci, "cv": sd / m if n > 1 and m else float("nan"), "min": min(xs), "max": max(xs)}


def check_comparable(runs: List[dict], allow_quick: bool = False) -> List[str]:
    """Return a list of reasons the runs must not be pooled (empty = comparable)."""
    problems = []
    if len(runs) < 2:
        problems.append("need at least 2 runs to estimate spread")
    for key in ("gpu", "torch", "cuda", "compute_capability"):
        vals = {r["meta"].get(key) for r in runs}
        if len(vals) > 1:
            problems.append(f"runs differ in {key}: {sorted(map(str, vals))}")
    if not allow_quick and any(r["meta"].get("quick") or r["meta"].get("iters", 0) < 10 for r in runs):
        problems.append("contains quick/smoke runs (iters < 10); pass --allow-quick to include them")
    return problems


def aggregate(runs: List[dict]) -> dict:
    sec = lambda r, k: r["sections"][k]["result"]  # noqa: E731
    out: dict = {"n_runs": len(runs), "gpu": runs[0]["meta"].get("gpu"), "torch": runs[0]["meta"].get("torch"), "cuda": runs[0]["meta"].get("cuda")}
    out["bandwidth_GBps"] = stat([sec(r, "bandwidth")["bandwidth_Bps"] / 1e9 for r in runs])
    out["peak_fp16_TFLOPS"] = stat([sec(r, "peak")["peak_flops"]["fp16"] / 1e12 for r in runs])
    gemm: Dict[int, List[float]] = {}
    for r in runs:
        for row in sec(r, "gemm_family")["rows"]:
            gemm.setdefault(row["batch"], []).append(row["ratio"])
    out["gemm_measured_over_predicted_by_batch"] = {str(b): stat(v) for b, v in sorted(gemm.items()) if len(v) == len(runs)}
    kv: Dict[int, Dict[str, List[float]]] = {}
    for r in runs:
        for row in sec(r, "kv_transfer")["rows"]:
            d = kv.setdefault(row["ctx"], {"h2d": [], "d2h": []})
            d["h2d"].append(row["h2d_Bps"] / 1e9)
            d["d2h"].append(row["d2h_Bps"] / 1e9)
    out["kv_transfer_GBps_by_ctx"] = {str(c): {k: stat(v) for k, v in d.items()} for c, d in sorted(kv.items()) if len(d["h2d"]) == len(runs)}
    out["stability_note"] = ("coefficient of variation (cv) above 0.05 for bandwidth or peak means the GPU clocks/thermals or neighbours are not stable enough to quote one number; "
                             "report the interval, not the mean alone")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--allow-quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = [json.load(open(f, encoding="utf-8")) for f in a.files]
    problems = check_comparable(runs, a.allow_quick)
    if problems:
        print("Refusing to aggregate:\n  - " + "\n  - ".join(problems))
        return 2
    res = aggregate(runs)
    b, p = res["bandwidth_GBps"], res["peak_fp16_TFLOPS"]
    print(f"{res['gpu']}: {res['n_runs']} runs\n  bandwidth {b['mean']:.1f} +- {b['ci95']:.1f} GB/s (cv {b['cv']:.3f})\n  fp16 peak {p['mean']:.2f} +- {p['ci95']:.2f} TFLOPS (cv {p['cv']:.3f})")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
