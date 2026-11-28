"""Pure arithmetic for the GPU measurement (no torch, no GPU): the same roofline prediction used for the CPU run,
plus byte counts for decode attention and KV transfer. Kept separate so it can be unit-tested anywhere.

Model (identical to experiments/measure_roofline.py, with an element size `b` instead of a fixed 4 bytes):
    time(B)  = max(flops(B) / peak, bytes(B) / bandwidth)
    flops    = 2 * M * K * B
    bytes    = b * (M*K + K*B + M*B)
"""
from calibration.hardware_models import MODELS, kv_bytes_per_token


def gemm_flops(m: int, k: int, batch: int) -> float:
    return 2.0 * m * k * batch


def gemm_bytes(m: int, k: int, batch: int, elem: int) -> float:
    return float(elem) * (m * k + k * batch + m * batch)


def predicted_time(flops: float, nbytes: float, peak_flops: float, bandwidth: float) -> float:
    return max(flops / peak_flops, nbytes / bandwidth)


def balance_batch(m: int, k: int, elem: int, peak_flops: float, bandwidth: float):
    """Batch B* where compute time equals memory time; None when the machine is memory-bound at every batch (denominator <= 0)."""
    denom = 2.0 * m * k / peak_flops - elem * (k + m) / bandwidth
    if denom <= 0:
        return None
    return (elem * m * k / bandwidth) / denom


def first_batch_at_fraction(rows, peak_flops: float, fraction: float = 0.5):
    """First measured batch whose achieved FLOP/s reaches `fraction` of the measured peak, or None."""
    for r in rows:
        if r["achieved_flops"] >= fraction * peak_flops:
            return r["batch"]
    return None


def attn_decode_bytes(batch: int, ctx: int, kv_heads: int, head_dim: int, elem: int) -> float:
    """Bytes read by one decode-step attention for `batch` sequences of `ctx` tokens: K and V (the query and output are negligible)."""
    return 2.0 * batch * ctx * kv_heads * head_dim * elem


def kv_transfer_bytes(ctx: int, model: str = "Llama-3.1-8B", elem: int = 2) -> int:
    """KV cache of ONE sequence across all layers, as moved by an offload/prefetch."""
    return int(ctx * kv_bytes_per_token(MODELS[model], elem))


def ratio_row(measured_s: float, predicted_s: float) -> float:
    return measured_s / predicted_s if predicted_s > 0 else float("inf")


def decode_steps_equivalent(transfer_s: float, step_s: float) -> float:
    """How many decode steps fit in one KV fetch: the quantity the H13 threshold story is about."""
    return transfer_s / step_s if step_s > 0 else float("inf")
