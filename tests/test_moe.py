import math

from calibration.moe_model import (MOE_MODELS, active_params, critical_batch, dense_equivalent, max_batch_in_memory, step, total_params,
                                   union_skewed, union_uniform)

Q, X = MOE_MODELS["Qwen3-30B-A3B"], MOE_MODELS["Mixtral-8x7B"]
DEV = "H100 SXM"


def test_parameter_counts_match_published_sizes():
    # published: Qwen3-30B-A3B 30.5B total / 3.3B active (incl. 0.31B embedding); Mixtral-8x7B 46.7B total / 12.9B active (incl. 0.13B embedding)
    assert abs(total_params(Q) / 1e9 - 30.5) < 0.1 and abs(total_params(X) / 1e9 - 46.7) < 0.1
    assert abs((active_params(Q) + Q["hidden"] * Q["vocab"]) / 1e9 - 3.3) < 0.1
    assert abs((active_params(X) + X["hidden"] * X["vocab"]) / 1e9 - 12.9) < 0.15


def test_union_limits_and_shape():
    for m in (Q, X):
        e, k = m["experts"], m["topk"]
        assert math.isclose(union_uniform(1, e, k), k)               # one token touches exactly k experts
        assert union_uniform(0, e, k) == 0
        vals = [union_uniform(n, e, k) for n in range(1, 60)]
        assert all(b > a for a, b in zip(vals, vals[1:])) and vals[-1] <= e + 1e-9      # strictly increasing, bounded by E
        assert all((c - b) <= (b - a) + 1e-12 for a, b, c in zip(vals, vals[1:], vals[2:]))   # concave
    assert abs(union_uniform(8, 128, 8) / 128 - 0.403) < 0.001       # the 8-position case quoted from the Hugging Face blog


def test_skew_only_lowers_the_union_and_zero_skew_matches_uniform():
    for n in (1, 8, 64):
        assert math.isclose(union_skewed(n, 128, 8, 0.0), union_uniform(n, 128, 8), rel_tol=1e-6)
        assert union_skewed(n, 128, 8, 1.0) <= union_uniform(n, 128, 8) + 1e-9
    for n in (2, 8, 64):
        assert union_skewed(n, 128, 8, 1.0) < union_uniform(n, 128, 8)
    assert math.isclose(union_skewed(1, 128, 8, 1.0), 8.0, rel_tol=1e-6)       # a single token still touches k experts


def test_dense_equivalent_reads_everything_and_has_same_compute():
    d = dense_equivalent(Q)
    assert abs(active_params(d) / active_params(Q) - 1) < 0.01                       # differs only by the (tiny) router
    tc_m, tm_m = step(Q, DEV, 16, 16, "bf16", 1, 4096)
    tc_d, tm_d = step(d, DEV, 16, 16, "bf16", 1, 4096)
    assert math.isclose(tc_m, tc_d, rel_tol=0.01) and math.isclose(tm_m, tm_d, rel_tol=0.02)      # at one token the MoE reads about as much as the dense twin (router differs)
    _, tm_m64 = step(Q, DEV, 16, 16, "bf16", 64, 4096)
    _, tm_d64 = step(d, DEV, 16, 16, "bf16", 64, 4096)
    assert tm_m64 > tm_d64                                                         # many tokens: the MoE reads more experts than the dense twin


def test_critical_batch_is_consistent_and_moe_needs_more_batch_than_dense_twin():
    for spec in (4, 8):
        b = critical_batch(Q, DEV, 16, 16, "bf16", 512, spec)
        bd = critical_batch(dense_equivalent(Q), DEV, 16, 16, "bf16", 512, spec)
        assert b is not None and bd is not None and b > bd
        tc, tm = step(Q, DEV, 16, 16, "bf16", b, 512, spec)
        assert math.isclose(tc, tm, rel_tol=1e-6)                                  # at B* compute time equals memory time


def test_speculation_lowers_the_batch_needed_for_balance():
    assert critical_batch(Q, DEV, 16, 16, "bf16", 512, spec=1) is None               # plain decode at this context: KV reads alone exceed the compute, no balance
    b1 = critical_batch(Q, DEV, 16, 16, "bf16", 512, spec=4)
    b8 = critical_batch(Q, DEV, 16, 16, "bf16", 512, spec=8)
    assert b8 < b1                                                                  # more tokens per sequence reach balance with fewer sequences


def test_skew_does_not_move_balance_point_because_union_is_saturated_there():
    b0 = critical_batch(Q, DEV, 16, 16, "bf16", 512, spec=8)
    b1 = critical_batch(Q, DEV, 16, 16, "bf16", 512, spec=8, skew=1.0)
    assert b1 <= b0 + 1e-9 and math.isclose(b1, b0, rel_tol=1e-3)               # at B* nearly every expert is read, so popularity skew barely matters
    assert union_uniform(b0 * 8, 128, 8) / 128 > 0.999


def test_balance_factor_is_total_over_active_params():
    for m in (Q, X):
        f = critical_batch(m, DEV, 16, 16, "bf16", 512, 8) / critical_batch(dense_equivalent(m), DEV, 16, 16, "bf16", 512, 8)
        assert math.isclose(f, total_params(m) / active_params(m), rel_tol=0.12)  # when all experts are read the weight time scales with total / active


def test_memory_fit():
    assert max_batch_in_memory(X, DEV, 16, 16, 4096) is None                        # 93 GB of bf16 weights do not fit 80 GB
    assert max_batch_in_memory(X, DEV, 8, 8, 4096) > 0
