import pytest

from calibration.hardware_models import DEVICES, MODELS, dense_peak, head_dim, kv_bytes_per_token, matmul_params, total_params
from experiments.h6_calibrated import SCENARIOS, critical_batch, step


def test_dense_peak_is_half_the_printed_sparse_figure():
    assert dense_peak("H100 SXM", "bf16") == pytest.approx(1979.0e12 / 2)


def test_architecture_derivations():
    m = MODELS["Llama-3.1-8B"]
    assert head_dim(m) == 128
    assert kv_bytes_per_token(m, 2) == 2 * 32 * 8 * 128 * 2
    assert total_params(m) > matmul_params(m) > 7e9


@pytest.mark.parametrize("device", list(DEVICES))
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_batch_one_decode_is_memory_bound_and_closed_form_matches_the_crossing(device, model, scenario):
    _, wb, kvb, fmt = scenario
    tc, tm = step(model, device, wb, kvb, fmt, 1, 512)
    assert tc < tm
    b = critical_batch(model, device, wb, kvb, fmt, 512)
    if b is not None:
        tc2, tm2 = step(model, device, wb, kvb, fmt, b, 512)
        assert tc2 == pytest.approx(tm2, rel=1e-9)


def test_prefill_ratio_increases_with_prompt_length_and_has_a_balance_point():
    from experiments.h6_calibrated import prefill_balance, prefill_ratio
    r = [prefill_ratio("Llama-3.1-8B", "H100 SXM", 16, 16, "bf16", s) for s in (16, 128, 1024, 8192)]
    assert r == sorted(r) and r[0] < 1 < r[-1]
    s_star = prefill_balance("Llama-3.1-8B", "H100 SXM", 16, 16, "bf16")
    assert prefill_ratio("Llama-3.1-8B", "H100 SXM", 16, 16, "bf16", s_star) >= 1 > prefill_ratio("Llama-3.1-8B", "H100 SXM", 16, 16, "bf16", s_star - 1)
