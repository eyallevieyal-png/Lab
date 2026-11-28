"""Measured roofline on a real GPU (free-tier Colab/Kaggle T4 is enough). Companion to measure_roofline.py (CPU).

What it measures (all timed with CUDA events, after warmup):
  1. memory bandwidth      best of device-to-device copy and a full-array reduction (fp32, ~1 GiB)
  2. peak matmul           best of large fp16 (tensor core) and fp32 GEMMs
  3. decode-like family    Y = W @ X, W 8192x8192 fp16, X 8192xB, B = 1..512; compared with time = max(flops/peak, bytes/bw)
  4. decode attention      one query token against a KV cache of B sequences x ctx tokens (Llama-3.1-8B shape: 32 q heads, 8 kv heads, d=128)
                           tests the claim "the KV cache is read per sequence, so attention stays memory-bound at any batch"
  5. KV transfer           pinned host<->device copy of the KV cache of ONE Llama-3.1-8B sequence (bf16, all layers) at several contexts

Limits (stated, not hidden): one GPU, one run; the GPU is probably a T4, not an H100, so absolute numbers do not transfer, only the
SHAPE of the curves and the ratio measured/predicted; clocks can throttle (recorded before and after); the attention kernel chosen by PyTorch
on this GPU is recorded but not controlled; section 5 measures an idle link, not the contended link of the H13 model.

Run (from the repo root):   python -m experiments.measure_gpu            full run, a few minutes
                            python -m experiments.measure_gpu --quick    short smoke run first (recommended)
Writes experiments/results/measured_gpu_<gpu>.json. Needs only PyTorch with CUDA (preinstalled on Colab/Kaggle). No network, no secrets.
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

from experiments import gpu_bench_lib as L

try:
    import torch
except ImportError:  # pragma: no cover
    sys.exit("PyTorch is not installed. On Colab/Kaggle it is preinstalled; locally: https://pytorch.org/get-started/locally/")


def gpu_state():
    """Clocks, temperature and power right now (nvidia-smi); empty dict when unavailable. Used to show throttling."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0]
        keys = ["sm_clock_mhz", "sm_clock_max_mhz", "temp_c", "power_w", "power_limit_w"]
        return dict(zip(keys, [x.strip() for x in out.split(",")]))
    except Exception:
        return {}


def timed(fn, warmup: int, iters: int):
    """(min, median) seconds of fn() measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return min(ts), statistics.median(ts)


def free_all():
    torch.cuda.empty_cache()


def section(name, fn, report):
    """Run one section; a failure (e.g. out of memory) is recorded and does not stop the others."""
    print(f"\n== {name}")
    t0 = time.time()
    before = gpu_state()
    try:
        res = fn()
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("  FAILED:", res["error"])
    free_all()
    report[name] = {"result": res, "gpu_before": before, "gpu_after": gpu_state(), "seconds": round(time.time() - t0, 1)}
    return res


def measure_bandwidth(args):
    n = (1 << 28) if not args.quick else (1 << 26)            # 1 GiB / 256 MiB of fp32
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    t_copy, _ = timed(lambda: y.copy_(x), 3, args.iters)
    t_sum, _ = timed(lambda: x.sum(), 3, args.iters)
    bw_copy = 2.0 * x.numel() * 4 / t_copy                    # one read plus one write
    bw_sum = x.numel() * 4 / t_sum                            # one read
    bw = max(bw_copy, bw_sum)
    print(f"  copy {bw_copy / 1e9:.1f} GB/s   reduce {bw_sum / 1e9:.1f} GB/s   -> using {bw / 1e9:.1f} GB/s")
    return {"bandwidth_Bps": bw, "copy_Bps": bw_copy, "reduce_Bps": bw_sum, "array_bytes": x.numel() * 4}


def measure_peak(args):
    n = 8192 if not args.quick else 4096
    out = {}
    for name, dt in (("fp16", torch.float16), ("fp32", torch.float32)):
        a = torch.randn(n, n, device="cuda", dtype=dt) * 0.05
        b = torch.randn(n, n, device="cuda", dtype=dt) * 0.05
        t, _ = timed(lambda: a @ b, 2, max(3, args.iters // 2))
        out[name] = 2.0 * n ** 3 / t
        print(f"  {name} matmul {n}x{n}: {out[name] / 1e12:.2f} TFLOPS")
        del a, b
        free_all()
    return {"peak_flops": out, "n": n}


def measure_family(args, bw, peak):
    m = k = 8192 if not args.quick else 4096
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if not args.quick else [1, 8, 64, 512]
    elem = 2
    w = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.05
    rows = []
    print(f"  {'B':>5} {'meas ms':>9} {'pred ms':>9} {'meas/pred':>10} {'TFLOPS':>8}")
    for b in batches:
        x = torch.randn(k, b, device="cuda", dtype=torch.float16) * 0.05
        t, med = timed(lambda: w @ x, 5, args.iters)
        fl, nb = L.gemm_flops(m, k, b), L.gemm_bytes(m, k, b, elem)
        pred = L.predicted_time(fl, nb, peak, bw)
        rows.append({"batch": b, "measured_s": t, "median_s": med, "predicted_s": pred, "ratio": L.ratio_row(t, pred),
                     "achieved_flops": fl / t, "intensity_flop_per_byte": fl / nb})
        print(f"  {b:5d} {t * 1e3:9.3f} {pred * 1e3:9.3f} {L.ratio_row(t, pred):10.2f} {fl / t / 1e12:8.2f}")
    return {"m": m, "k": k, "dtype": "fp16", "rows": rows, "predicted_balance_batch": L.balance_batch(m, k, elem, peak, bw),
            "measured_batch_at_half_peak": L.first_batch_at_fraction(rows, peak)}


def measure_attention(args, bw):
    import torch.nn.functional as F
    q_heads, kv_heads, d, elem = 32, 8, 128, 2
    ctxs = [512, 4096, 16384] if not args.quick else [512, 4096]
    batches = [1, 4, 16, 64] if not args.quick else [1, 16]
    rows, gqa_ok = [], True
    for ctx in ctxs:
        for b in batches:
            nbytes = L.attn_decode_bytes(b, ctx, kv_heads, d, elem)
            # a quarter of the cap: some PyTorch kernels expand grouped KV heads to full heads internally (4x memory)
            if nbytes > args.max_kv_gib * 2 ** 30 / 4:
                rows.append({"ctx": ctx, "batch": b, "skipped": "KV larger than a quarter of --max-kv-gib"})
                continue
            try:
                q = torch.randn(b, q_heads, 1, d, device="cuda", dtype=torch.float16)
                kk = torch.randn(b, kv_heads, ctx, d, device="cuda", dtype=torch.float16)
                vv = torch.randn(b, kv_heads, ctx, d, device="cuda", dtype=torch.float16)
                try:
                    t, med = timed(lambda: F.scaled_dot_product_attention(q, kk, vv, enable_gqa=True), 3, args.iters)
                except TypeError:
                    gqa_ok = False   # old PyTorch: fall back to the full-head shape; recorded, bytes then use 32 kv heads
                    kk, vv = kk.repeat_interleave(q_heads // kv_heads, 1), vv.repeat_interleave(q_heads // kv_heads, 1)
                    nbytes = L.attn_decode_bytes(b, ctx, q_heads, d, elem)
                    t, med = timed(lambda: F.scaled_dot_product_attention(q, kk, vv), 3, args.iters)
            except RuntimeError as e:   # out of memory or an unsupported kernel: record this case, keep the others
                rows.append({"ctx": ctx, "batch": b, "error": f"{type(e).__name__}: {str(e)[:120]}"})
                print(f"  ctx {ctx:6d} B {b:3d}: FAILED {type(e).__name__}")
                free_all()
                continue
            rows.append({"ctx": ctx, "batch": b, "bytes": nbytes, "measured_s": t, "median_s": med, "predicted_mem_s": nbytes / bw,
                         "ratio": L.ratio_row(t, nbytes / bw), "effective_Bps": nbytes / t})
            print(f"  ctx {ctx:6d} B {b:3d}: {t * 1e3:8.3f} ms  mem-floor {nbytes / bw * 1e3:8.3f} ms  ratio {L.ratio_row(t, nbytes / bw):5.2f}")
            del q, kk, vv
            free_all()
    return {"rows": rows, "gqa_native": gqa_ok, "note": "ratio = measured / (KV bytes / measured bandwidth); the kernel PyTorch picked is not controlled"}


def measure_kv_transfer(args):
    ctxs = [512, 4096, 32768] if not args.quick else [512]
    rows = []
    for ctx in ctxs:
        nbytes = L.kv_transfer_bytes(ctx)
        if nbytes > args.max_kv_gib * 2 ** 30:
            rows.append({"ctx": ctx, "bytes": nbytes, "skipped": "larger than --max-kv-gib"})
            continue
        host = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
        dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        t_h2d, _ = timed(lambda: dev.copy_(host, non_blocking=True), 2, max(3, args.iters // 2))
        t_d2h, _ = timed(lambda: host.copy_(dev, non_blocking=True), 2, max(3, args.iters // 2))
        rows.append({"ctx": ctx, "bytes": nbytes, "h2d_s": t_h2d, "d2h_s": t_d2h, "h2d_Bps": nbytes / t_h2d, "d2h_Bps": nbytes / t_d2h})
        print(f"  ctx {ctx:6d} ({nbytes / 2 ** 20:7.0f} MiB): H2D {t_h2d * 1e3:8.2f} ms ({nbytes / t_h2d / 1e9:5.1f} GB/s)   D2H {t_d2h * 1e3:8.2f} ms")
        del host, dev
        free_all()
    return {"rows": rows, "model": "Llama-3.1-8B", "dtype": "bf16 (2 bytes)", "link": "idle; no concurrent traffic"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes and few repeats: a smoke test of the script itself")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--max-kv-gib", type=float, default=4.0, help="skip attention / transfer cases whose KV would exceed this (protects small GPUs and host RAM)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.quick:
        args.iters = min(args.iters, 5)
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible. On Colab: Runtime > Change runtime type > T4 GPU. On Kaggle: Settings > Accelerator > GPU T4.")
    torch.manual_seed(0)
    prop = torch.cuda.get_device_properties(0)
    meta = {"gpu": prop.name, "gpu_memory_GiB": round(prop.total_memory / 2 ** 30, 2), "sm_count": prop.multi_processor_count,
            "compute_capability": f"{prop.major}.{prop.minor}", "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "quick": args.quick, "iters": args.iters, "max_kv_gib": args.max_kv_gib,
            "started_unix": int(time.time())}
    print(f"GPU: {prop.name}, {meta['gpu_memory_GiB']} GiB, torch {torch.__version__}, CUDA {torch.version.cuda}")
    report = {}
    bw = section("bandwidth", lambda: measure_bandwidth(args), report)
    pk = section("peak", lambda: measure_peak(args), report)
    if isinstance(bw, dict) and "bandwidth_Bps" in bw and isinstance(pk, dict) and "peak_flops" in pk:
        section("gemm_family", lambda: measure_family(args, bw["bandwidth_Bps"], pk["peak_flops"]["fp16"]), report)
        section("decode_attention", lambda: measure_attention(args, bw["bandwidth_Bps"]), report)
    else:
        print("\nSkipping the dependent sections: bandwidth or peak failed.")
    section("kv_transfer", lambda: measure_kv_transfer(args), report)
    out = {"meta": meta, "sections": report}
    slug = re.sub(r"[^A-Za-z0-9]+", "_", prop.name).strip("_").lower()
    path = args.out or os.path.join("experiments", "results", f"measured_gpu_{slug}{'_quick' if args.quick else ''}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
