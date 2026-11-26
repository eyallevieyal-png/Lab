"""Measured roofline of the machine this runs on, and a test of the model's prediction (real measurement, not a simulation).

1. memory bandwidth: best of several reads of a large float32 array (sum and dot).
2. peak compute: best of several large float32 matrix multiplications.
3. the decode-like family  Y = W @ X  with W (M x K) read once and X (K x B): B = 1 is a matrix-vector product (decode-like),
   large B is a matrix-matrix product (prefill/large-batch-like). The model predicts
       time(B) = max(flops(B) / peak, bytes(B) / bandwidth),   flops = 2*M*K*B,   bytes = 4*(M*K + K*B + M*B)
   and the batch B* where compute time equals memory time. We compare predictions with measurements.

Limits: one CPU, float32 only, BLAS thread scheduling and caches are not modelled, other processes add noise. Results are machine-specific.
Run: python experiments/measure_roofline.py   (writes experiments/results/measured_roofline.json)
"""
import json
import os
import platform
import time

import numpy as np

M = K = 8192          # W is 256 MiB in float32
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def best(fn, repeats=5):
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return min(t), sorted(t)[len(t) // 2]


def main():
    rng = np.random.default_rng(0)
    # 1. bandwidth
    a = rng.random(2 ** 27, dtype=np.float32)       # 512 MiB
    b = rng.random(2 ** 26, dtype=np.float32)       # 256 MiB
    t_sum, _ = best(lambda: a.sum())
    bw_sum = a.nbytes / t_sum
    t_dot, _ = best(lambda: np.dot(b, b))
    bw_dot = b.nbytes / t_dot                        # dot reads b twice but from the same array; count one read of 256 MiB
    bw = max(bw_sum, bw_dot * 1.0)
    del a, b
    # 2. peak compute
    n = 2048
    A = rng.random((n, n), dtype=np.float32)
    B = rng.random((n, n), dtype=np.float32)
    t_mm, _ = best(lambda: A @ B, 7)
    peak = 2 * n ** 3 / t_mm
    del A, B
    print(f"measured bandwidth: {bw / 1e9:.1f} GB/s (sum {bw_sum / 1e9:.1f}, dot {bw_dot / 1e9:.1f});  peak float32 matmul: {peak / 1e9:.1f} GFLOPS;  ridge {peak / bw:.1f} FLOP/byte")

    # 3. the family
    W = rng.random((M, K), dtype=np.float32)
    rows = []
    print(f"\n{'B':>5} {'measured ms':>12} {'predicted ms':>13} {'meas/pred':>10} {'t_mem/measured':>15} {'GFLOPS':>8}")
    for batch in BATCHES:
        X = rng.random((K, batch), dtype=np.float32)
        t, _ = best(lambda: W @ X, 5 if batch < 256 else 3)
        flops = 2.0 * M * K * batch
        nbytes = 4.0 * (M * K + K * batch + M * batch)
        pred = max(flops / peak, nbytes / bw)
        rows.append({"batch": batch, "measured_s": t, "predicted_s": pred, "ratio": t / pred, "memory_time_share": (nbytes / bw) / t,
                     "achieved_gflops": flops / t / 1e9, "intensity_flop_per_byte": flops / nbytes})
        print(f"{batch:5d} {t * 1e3:12.2f} {pred * 1e3:13.2f} {t / pred:10.2f} {(nbytes / bw) / t:15.2f} {flops / t / 1e9:8.1f}")
    # balance batch: predicted (closed form) and measured (first B where achieved GFLOPS >= 50% of peak)
    b_pred = (4.0 * M * K / bw) / (2.0 * M * K / peak - 4.0 * (K + M) / bw)
    b_meas = next((r["batch"] for r in rows if r["achieved_gflops"] * 1e9 >= 0.5 * peak), None)
    print(f"\npredicted balance batch B* = {b_pred:.1f};  first measured batch reaching 50% of peak = {b_meas}")
    out = {"cpu": platform.processor(), "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__,
           "bandwidth_GBps": bw / 1e9, "peak_gflops": peak / 1e9, "ridge": peak / bw, "predicted_balance_batch": b_pred,
           "measured_batch_at_half_peak": b_meas, "rows": rows}
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/measured_roofline.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
