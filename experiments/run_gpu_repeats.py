"""Run experiments/measure_gpu.py N times in one session and aggregate the runs (needs a CUDA GPU; on Colab: Runtime > T4 / A100 / H100).

    python experiments/run_gpu_repeats.py --runs 5 --cooldown 20
Writes experiments/results/repeats/<gpu>/run_<i>.json and aggregated.json. Nothing is sent anywhere; no secrets are used.
Run the repeats in ONE session but separate processes: that exposes run-to-run variation (clocks, thermals, allocator state) that a single process hides.
Do not quote a hardware number from fewer than 3 full runs. Record `nvidia-smi -q -d CLOCK,POWER` next to the result if the GPU throttles (the script stores clocks per section).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=20.0, help="seconds between runs so the GPU can cool down")
    ap.add_argument("--max-kv-gib", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    probe = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        sys.exit("No NVIDIA GPU visible (nvidia-smi failed). This script must run on the GPU machine.")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", probe.stdout.splitlines()[0]).strip("_").lower()
    outdir = os.path.join(ROOT, "experiments", "results", "repeats", slug)
    os.makedirs(outdir, exist_ok=True)
    files = []
    for i in range(a.runs):
        path = os.path.join(outdir, f"run_{i + 1}.json")
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "measure_gpu.py"), "--iters", str(a.iters), "--max-kv-gib", str(a.max_kv_gib), "--out", path]).returncode
        if rc != 0:
            sys.exit(f"run {i + 1} failed (exit {rc}); not aggregating")
        files.append(path)
        if i + 1 < a.runs:
            time.sleep(a.cooldown)
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "experiments.aggregate_gpu_repeats", *files, "--out", os.path.join(outdir, "aggregated.json")], cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
