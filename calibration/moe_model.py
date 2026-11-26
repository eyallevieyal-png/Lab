"""Mixture-of-experts extension of the roofline model (compute only, no measurement).

Dense decode reads every weight once per step. An MoE reads only the experts that at least one token of the step is routed to, so the weight
bytes grow with the number of tokens n in the step while the compute is always n * (active parameters). Under independent uniform routing
(each token picks k of E experts uniformly, an ASSUMPTION; real routers are skewed and correlated) the expected number of distinct experts per layer is

    U(n) = E * (1 - (1 - k/E)^n),     U(1) = k,     U(n) -> E.

A token step here is n = batch * spec, where spec is the number of positions verified per sequence (speculative decoding; spec = 1 is plain decode).
Architecture values are the public config.json files (read 2026-11-26, see SOURCES). Nothing here is measured by us.
"""
from __future__ import annotations

from calibration.hardware_models import DEVICES, dense_peak

SOURCES = {
    "qwen3_moe": "https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json",
    "mixtral": "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json (head_dim derived as hidden / heads, as for the dense models)",
}

MOE_MODELS = {
    "Qwen3-30B-A3B": {"hidden": 2048, "layers": 48, "heads": 32, "kv_heads": 4, "head_dim": 128, "vocab": 151936, "experts": 128, "topk": 8, "expert_inter": 768},
    "Mixtral-8x7B": {"hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 8, "head_dim": 128, "vocab": 32000, "experts": 8, "topk": 2, "expert_inter": 14336},
}


def expert_params(m: dict) -> int:
    return 3 * m["hidden"] * m["expert_inter"]


def attn_params(m: dict) -> int:
    h, hd = m["hidden"], m["head_dim"]
    return 2 * h * m["heads"] * hd + 2 * h * m["kv_heads"] * hd          # q, o, k, v projections


def nonexpert_params(m: dict) -> int:
    """Parameters read every step whatever the routing: attention, router and the output head (embedding lookup excluded: only rows are read)."""
    return m["layers"] * (attn_params(m) + m["hidden"] * m["experts"]) + m["hidden"] * m["vocab"]


def active_params(m: dict) -> int:
    """Parameters that take part in the matmuls of one token."""
    return nonexpert_params(m) + m["layers"] * m["topk"] * expert_params(m)


def total_params(m: dict) -> int:
    return nonexpert_params(m) + m["layers"] * m["experts"] * expert_params(m) + m["hidden"] * m["vocab"]   # + input embedding table


def kv_bytes_per_token(m: dict, bytes_per_elem: float) -> float:
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * bytes_per_elem


def attn_flops_per_token(m: dict, context: int) -> float:
    return 4.0 * m["layers"] * m["heads"] * m["head_dim"] * context


def dense_equivalent(m: dict) -> dict:
    """Control model with the same active parameters per token but no sparsity: every expert is always read (E = k)."""
    d = dict(m)
    d["experts"] = m["topk"]
    return d


def union_uniform(n: float, experts: int, topk: int) -> float:
    """Expected distinct experts per layer touched by n tokens under independent uniform routing."""
    return experts * (1.0 - (1.0 - topk / experts) ** n)


def union_skewed(n: float, experts: int, topk: int, skew: float) -> float:
    """Same with popularity w_i ~ i^-skew (skew 0 = uniform). A token includes expert i with probability q_i = min(1, c * w_i), c chosen so that
    sum q_i = topk. This is an APPROXIMATION (it ignores that a token's k picks are distinct draws); it only bounds how much skew can lower the union."""
    w = [(i + 1) ** -skew for i in range(experts)]
    lo, hi = 0.0, 1e9
    for _ in range(200):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if sum(min(1.0, c * x) for x in w) < topk else (lo, c)
    q = [min(1.0, hi * x) for x in w]
    return sum(1.0 - (1.0 - qi) ** n for qi in q)


def step(m: dict, device: str, wbits: int, kvbits: int, fmt: str, batch: float, context: int, spec: int = 1, skew: float = 0.0):
    """(t_compute, t_memory) in seconds of one decode step: `batch` sequences, `spec` positions each, `context` cached tokens per sequence."""
    n = batch * spec
    flops = n * (2 * active_params(m) + attn_flops_per_token(m, context))
    union = union_uniform(n, m["experts"], m["topk"]) if skew == 0 else union_skewed(n, m["experts"], m["topk"], skew)
    nbytes = (nonexpert_params(m) + m["layers"] * union * expert_params(m)) * wbits / 8 + batch * context * kv_bytes_per_token(m, kvbits / 8)
    return flops / dense_peak(device, fmt), nbytes / (DEVICES[device]["bandwidth_TBps"] * 1e12)


def critical_batch(m: dict, device: str, wbits: int, kvbits: int, fmt: str, context: int, spec: int = 1, skew: float = 0.0, cap: float = float(1 << 24)):
    """Smallest batch (sequences) where t_compute >= t_memory, or None. g(B) = t_c - t_m is convex (linear minus concave) and negative at 0, so it crosses zero at most once."""
    def g(b):
        tc, tm = step(m, device, wbits, kvbits, fmt, b, context, spec, skew)
        return tc - tm
    if g(cap) < 0:
        return None
    lo, hi = 0.0, 1.0
    while g(hi) < 0:
        lo, hi = hi, hi * 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    return hi


def max_batch_in_memory(m: dict, device: str, wbits: int, kvbits: int, context: int):
    """Largest batch whose weights (all experts) plus KV cache fit in one device (None if the weights alone do not fit)."""
    free = DEVICES[device]["memory_GB"] * 1e9 - total_params(m) * wbits / 8
    return None if free <= 0 else int(free // (context * kv_bytes_per_token(m, kvbits / 8)))
