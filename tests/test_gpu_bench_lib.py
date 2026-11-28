import pytest

from experiments import gpu_bench_lib as L


def test_gemm_counts():
    assert L.gemm_flops(8192, 8192, 1) == 2.0 * 8192 * 8192
    assert L.gemm_bytes(4, 4, 1, 2) == 2 * (16 + 4 + 4)


def test_prediction_is_the_larger_of_compute_and_memory_time():
    assert L.predicted_time(flops=1e12, nbytes=1e9, peak_flops=1e12, bandwidth=1e11) == pytest.approx(1.0)   # compute-bound
    assert L.predicted_time(flops=1e9, nbytes=1e11, peak_flops=1e12, bandwidth=1e11) == pytest.approx(1.0)   # memory-bound


def test_balance_batch_equalises_compute_and_memory_time():
    m = k = 8192
    peak, bw = 6.5e13, 3.2e11
    b = L.balance_batch(m, k, 2, peak, bw)
    assert b is not None and b > 1
    compute, memory = L.gemm_flops(m, k, b) / peak, L.gemm_bytes(m, k, b, 2) / bw
    assert compute == pytest.approx(memory, rel=1e-9)


def test_balance_batch_none_when_always_memory_bound():
    # B -> infinity is still memory-bound when peak >= m*bw/elem (here 4e15): the batch-independent traffic of W and Y keeps memory ahead
    assert L.balance_batch(8192, 8192, 2, peak_flops=1e18, bandwidth=1e12) is None
    assert L.balance_batch(8192, 8192, 2, peak_flops=1e9, bandwidth=1e12) is not None   # slow compute: a finite balance batch exists


def test_balance_batch_matches_the_cpu_script_formula_for_4_byte_elements():
    m = k = 8192
    peak, bw = 222e9, 13.7e9
    cpu = (4.0 * m * k / bw) / (2.0 * m * k / peak - 4.0 * (k + m) / bw)   # same closed form as experiments/measure_roofline.py
    assert L.balance_batch(m, k, 4, peak, bw) == pytest.approx(cpu)


def test_first_batch_at_fraction():
    rows = [{"batch": 1, "achieved_flops": 1e10}, {"batch": 8, "achieved_flops": 6e10}, {"batch": 64, "achieved_flops": 9e10}]
    assert L.first_batch_at_fraction(rows, peak_flops=1e11) == 8
    assert L.first_batch_at_fraction(rows, peak_flops=1e12) is None


def test_attention_bytes_scale_with_batch_and_context():
    base = L.attn_decode_bytes(1, 512, 8, 128, 2)
    assert base == 2 * 512 * 8 * 128 * 2
    assert L.attn_decode_bytes(4, 512, 8, 128, 2) == 4 * base          # per-sequence KV: no amortisation across the batch
    assert L.attn_decode_bytes(1, 1024, 8, 128, 2) == 2 * base


def test_kv_transfer_bytes_for_llama_8b():
    assert L.kv_transfer_bytes(1) == 2 * 32 * 8 * 128 * 2              # 131072 bytes per token, all layers, bf16
    assert L.kv_transfer_bytes(512) == 512 * 131072


def test_degenerate_ratios_do_not_crash():
    assert L.ratio_row(1.0, 0.0) == float("inf")
    assert L.decode_steps_equivalent(1.0, 0.0) == float("inf")
    assert L.decode_steps_equivalent(2.0, 1.0) == 2.0
