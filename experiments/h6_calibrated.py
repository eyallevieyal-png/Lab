"""H6 calibrated on public hardware and model specifications.

One decode step for a batch of B sequences with context T tokens each:
  FLOPs  = B * (2 * matmul_params + attention_flops(T))
  bytes  = weight_bytes (read once per step) + B * T * kv_bytes_per_token
  t_compute = FLOPs / dense_peak(compute format),   t_memory = bytes / bandwidth.
The compute/memory time ratio plays the role of k* in the simulator (ratio < 1: memory-dominated).
B* is the batch where the ratio reaches 1 (t_compute = t_memory); none exists if the attention part alone is memory-bound.
This is a first-order roofline model: no overlap effects, no kernel inefficiency, no communication.

Run: python experiments/h6_calibrated.py   (writes experiments/results/h6_calibrated.json)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration.hardware_models import (DEVICES, MODELS, SOURCES, attn_flops_per_token, dense_peak, kv_bytes_per_token,  # noqa: E402
                                          matmul_params, total_params)

# (name, weight bits, kv bits, compute format)
SCENARIOS = [
    ("bf16 weights/KV, bf16 compute", 16, 16, "bf16"),
    ("fp8 weights/KV, fp8 compute", 8, 8, "fp8"),
    ("int4 weights only, bf16 compute (dequantize)", 4, 16, "bf16"),
]
CONTEXTS = [512, 4096, 32768]


def step(model, device, wbits, kvbits, fmt, batch, context):
    m = MODELS[model]
    flops = batch * (2 * matmul_params(m) + attn_flops_per_token(m, context))
    nbytes = total_params(m) * wbits / 8 + batch * context * kv_bytes_per_token(m, kvbits / 8)
    t_c = flops / dense_peak(device, fmt)
    t_m = nbytes / (DEVICES[device]["bandwidth_TBps"] * 1e12)
    return t_c, t_m


def critical_batch(model, device, wbits, kvbits, fmt, context):
    """Closed form B* where t_c = t_m, or None when attention alone keeps it memory-bound."""
    m = MODELS[model]
    peak, bw = dense_peak(device, fmt), DEVICES[device]["bandwidth_TBps"] * 1e12
    f1 = (2 * matmul_params(m) + attn_flops_per_token(m, context)) / peak          # compute time per sequence
    kv1 = context * kv_bytes_per_token(m, kvbits / 8) / bw                          # KV read time per sequence
    w = total_params(m) * wbits / 8 / bw                                            # weight read time per step
    d = f1 - kv1
    return None if d <= 0 else w / d


def max_batch_in_memory(model, device, wbits, kvbits, context):
    """Largest batch whose weights plus KV cache fit in one device's memory (None if even the weights do not fit)."""
    m = MODELS[model]
    free = DEVICES[device]["memory_GB"] * 1e9 - total_params(m) * wbits / 8
    return None if free <= 0 else int(free // (context * kv_bytes_per_token(m, kvbits / 8)))


def main():
    out = {"sources": SOURCES, "params": {k: {"matmul": matmul_params(v), "total": total_params(v)} for k, v in MODELS.items()}, "rows": []}
    print("computed parameters (from config.json architecture):", {k: f"{v['total'] / 1e9:.2f}B" for k, v in out["params"].items()})
    for dev in DEVICES:
        print(f"\n=== {dev}: dense peak bf16 {dense_peak(dev, 'bf16') / 1e12:.0f} TFLOPS (derived), {DEVICES[dev]['bandwidth_TBps']} TB/s ===")
        print(f"{'model':14} {'scenario':46} {'ctx':>6} {'t_c/t_m @B=1':>13} {'B* (balanced)':>14} {'max B (fits)':>13}")
        for model in MODELS:
            for name, wb, kvb, fmt in SCENARIOS:
                for ctx in CONTEXTS:
                    tc, tm = step(model, dev, wb, kvb, fmt, 1, ctx)
                    b = critical_batch(model, dev, wb, kvb, fmt, ctx)
                    mb = max_batch_in_memory(model, dev, wb, kvb, ctx)
                    reach = "B* never" if b is None else ("weights do not fit" if mb is None else ("B* > memory" if b > mb else "B* reachable"))
                    out["rows"].append({"device": dev, "model": model, "scenario": name, "context": ctx, "ratio_b1": tc / tm,
                                        "critical_batch": b, "max_batch_fits": mb, "verdict": reach})
                    print(f"{model:14} {name:46} {ctx:6d} {tc / tm:13.4f} {('never' if b is None else f'{b:,.0f}'):>14} {('none' if mb is None else f'{mb:,}'):>13}")
    # H6 premise check: peak speed-up vs bytes shrink going from bf16 to fp8 (same device, weights and KV)
    d = "H100 SXM"
    speed = dense_peak(d, "fp8") / dense_peak(d, "bf16")
    shrink = 16 / 8
    print(f"\nbf16 -> fp8 on {d}: compute peak x{speed:.1f}, bytes /{shrink:.1f}: the balance point moves by a factor {speed / shrink:.2f} "
          f"(inversion in H6's sense needs peak speed-up greater than the bytes shrink).")
    out["bf16_to_fp8_factor"] = speed / shrink
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/h6_calibrated.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
