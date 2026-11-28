import json
from pathlib import Path

import pytest

from experiments import analyze_gpu

DATA = Path(__file__).resolve().parent.parent / "experiments" / "results" / "measured_gpu_tesla_t4.json"


@pytest.fixture(scope="module")
def res():
    return analyze_gpu.analyze(json.loads(DATA.read_text(encoding="utf-8")))


def test_numbers_quoted_in_the_article(res):
    assert res["bandwidth_GBps"] == pytest.approx(276.1, abs=0.1)
    assert res["peak_fp16_TFLOPS"] == pytest.approx(24.9, abs=0.1)
    lo, hi = res["gemm_memory_regime_ratio_range"]
    assert 1.39 <= lo and hi <= 1.46
    assert res["balance_batch_predicted"] == pytest.approx(92.3, abs=0.1)
    assert res["balance_batch_first_measured_at_half_peak"] == 128
    assert res["gemv_b1_frac_of_bandwidth"] == pytest.approx(0.92, abs=0.01)
    assert res["gemm_max_achieved_over_measured_peak"] > 1.0   # the measured peak is not a ceiling


def test_attention_and_transfer_claims(res):
    lo, hi = res["attention_big_effective_frac_of_bw"]
    assert 0.36 <= lo and hi <= 0.43
    assert res["attention_time_per_byte_scaling"][4096]["time_per_byte_ratio"] == pytest.approx(0.91, abs=0.01)
    assert all(0.9 <= r["h2d_GBps"] <= 1.3 for r in res["kv_transfer"])
    steps = res["kv_fetch_in_decode_steps_extrapolated"]
    assert steps[4096] == pytest.approx(6.9, abs=0.1)
