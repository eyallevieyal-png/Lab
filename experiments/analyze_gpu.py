"""Derived numbers from a measure_gpu.py result file. Pure arithmetic on the JSON: nothing here is a new measurement.

Run: python -m experiments.analyze_gpu [path]   (default experiments/results/measured_gpu_tesla_t4.json)
"""
import json
import sys

from experiments import gpu_bench_lib as L

DEFAULT = "experiments/results/measured_gpu_tesla_t4.json"
BIG_ATTENTION_BYTES = 256 * 2 ** 20     # below this a launch-latency floor dominates a ~0.05 ms kernel


def analyze(data: dict) -> dict:
    s = data["sections"]
    bw = s["bandwidth"]["result"]["bandwidth_Bps"]
    peak = s["peak"]["result"]["peak_flops"]["fp16"]
    fam = s["gemm_family"]["result"]
    out = {"gpu": data["meta"]["gpu"], "bandwidth_GBps": bw / 1e9, "peak_fp16_TFLOPS": peak / 1e12}

    wbytes = fam["m"] * fam["k"] * 2
    rows = fam["rows"]
    out["gemm"] = [{"batch": r["batch"], "ratio": r["ratio"], "frac_of_peak": r["achieved_flops"] / peak,
                    "weight_read_GBps": wbytes / r["measured_s"] / 1e9} for r in rows]
    mem_rows = [r for r in rows if 2 <= r["batch"] <= 64]
    out["gemm_memory_regime_ratio_range"] = (min(r["ratio"] for r in mem_rows), max(r["ratio"] for r in mem_rows))
    out["gemm_max_achieved_over_measured_peak"] = max(r["achieved_flops"] for r in rows) / peak
    out["balance_batch_predicted"] = fam["predicted_balance_batch"]
    out["balance_batch_first_measured_at_half_peak"] = fam["measured_batch_at_half_peak"]
    b1 = rows[0]
    out["gemv_b1_frac_of_bandwidth"] = wbytes / b1["measured_s"] / bw

    att = [r for r in s["decode_attention"]["result"]["rows"] if "measured_s" in r]
    big = [r for r in att if r["bytes"] >= BIG_ATTENTION_BYTES]
    out["attention_big_effective_frac_of_bw"] = (min(r["effective_Bps"] / bw for r in big), max(r["effective_Bps"] / bw for r in big))
    out["attention_big_ratio_range"] = (min(r["ratio"] for r in big), max(r["ratio"] for r in big))
    by_ctx = {}
    for r in att:
        by_ctx.setdefault(r["ctx"], []).append(r)
    scaling = {}
    for ctx, rs in by_ctx.items():      # time per byte relative to the smallest batch >= 4: flat means no amortisation across the batch
        rs = sorted((x for x in rs if x["batch"] >= 4), key=lambda x: x["batch"])
        if len(rs) >= 2:
            scaling[ctx] = {"batch_from": rs[0]["batch"], "batch_to": rs[-1]["batch"],
                            "time_per_byte_ratio": (rs[-1]["measured_s"] / rs[-1]["bytes"]) / (rs[0]["measured_s"] / rs[0]["bytes"])}
    out["attention_time_per_byte_scaling"] = scaling

    kv = [r for r in s["kv_transfer"]["result"]["rows"] if "h2d_s" in r]
    out["kv_transfer"] = [{"ctx": r["ctx"], "MiB": r["bytes"] / 2 ** 20, "h2d_GBps": r["h2d_Bps"] / 1e9, "d2h_GBps": r["d2h_Bps"] / 1e9,
                           "h2d_s": r["h2d_s"]} for r in kv]
    # decode-step proxy: an 8B bf16 model reads ~16 GB of weights per step. It does NOT fit on a T4, so this is an extrapolation
    # from the measured B=1 weight-read rate, not a measurement.
    step_proxy = 16.06e9 / (wbytes / b1["measured_s"])
    out["decode_step_proxy_s_extrapolated"] = step_proxy
    out["kv_fetch_in_decode_steps_extrapolated"] = {r["ctx"]: L.decode_steps_equivalent(r["h2d_s"], step_proxy) for r in kv}
    return out


def main(argv=None) -> int:
    path = (argv or sys.argv[1:] or [DEFAULT])[0]
    with open(path, encoding="utf-8") as fh:
        res = analyze(json.load(fh))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
