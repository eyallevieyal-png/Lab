"""Public specifications used to calibrate H6 (read on 2026-11-25; see SOURCES). Nothing here is measured by us.

Throughput figures on NVIDIA's page are marked "with sparsity". Dense throughput is taken as half of them (the standard 2:4
structured-sparsity convention): that division is OUR derivation, not a number on the page.
"""

SOURCES = {
    "h100": "https://www.nvidia.com/en-us/data-center/h100/ (Product Specifications table; throughput rows footnoted 'With sparsity')",
    "llama": "https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/raw/main/config.json and .../Meta-Llama-3.1-70B/raw/main/config.json",
    "not_obtained": "AMD MI300X page timed out twice; no MI300X figures are used. NVIDIA datasheet PDF is password protected.",
}

# peak tensor throughput as printed (with sparsity), TFLOPS or TOPS; bandwidth in TB/s
DEVICES = {
    "H100 SXM": {"bandwidth_TBps": 3.35, "memory_GB": 80, "sparse_peak_T": {"bf16": 1979.0, "fp8": 3958.0, "int8": 3958.0}},
    "H100 NVL": {"bandwidth_TBps": 3.9, "memory_GB": 94, "sparse_peak_T": {"bf16": 1671.0, "fp8": 3341.0, "int8": 3341.0}},
}

# architecture values as printed in config.json
MODELS = {
    "Llama-3.1-8B": {"hidden": 4096, "inter": 14336, "layers": 32, "heads": 32, "kv_heads": 8, "vocab": 128256},
    "Llama-3.1-70B": {"hidden": 8192, "inter": 28672, "layers": 80, "heads": 64, "kv_heads": 8, "vocab": 128256},
}


def dense_peak(device: str, fmt: str) -> float:
    """Dense peak in FLOP/s (derived: printed sparse figure / 2)."""
    return DEVICES[device]["sparse_peak_T"][fmt] / 2.0 * 1e12


def head_dim(m: dict) -> int:
    return m["hidden"] // m["heads"]  # derived (config.json has no head_dim)


def matmul_params(m: dict) -> int:
    """Parameters that take part in matrix multiplies per token (untied lm_head included, embedding lookup excluded)."""
    h, i, L, kv, hd = m["hidden"], m["inter"], m["layers"], m["kv_heads"], head_dim(m)
    attn = h * h + 2 * h * kv * hd + h * h
    mlp = 3 * h * i
    return L * (attn + mlp) + h * m["vocab"]


def total_params(m: dict) -> int:
    return matmul_params(m) + m["hidden"] * m["vocab"]  # + input embedding table (norms ignored, negligible)


def kv_bytes_per_token(m: dict, bytes_per_elem: float) -> float:
    return 2 * m["layers"] * m["kv_heads"] * head_dim(m) * bytes_per_elem


def attn_flops_per_token(m: dict, context: int) -> float:
    return 4.0 * m["layers"] * m["heads"] * head_dim(m) * context
