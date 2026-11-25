import pytest

from src.kv_prefetch import KVConfig, breakeven_accuracy, closed_form, mean_stall


def test_idle_link_reactive_stalls_exactly_one_transfer_and_perfect_prefetch_hides_it():
    cfg = KVConfig(u_bg=0.0, gamma=0.0)
    assert mean_stall(cfg, "reactive", 0.0) == pytest.approx(1.0)
    assert mean_stall(cfg, "predictive", 1.0) == pytest.approx(0.0, abs=1e-9)


def test_common_random_numbers_make_predictive_stall_nonincreasing_in_accuracy():
    cfg = KVConfig(u_bg=0.5, gamma=0.05)
    vals = [mean_stall(cfg, "predictive", p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_results_are_deterministic():
    cfg = KVConfig(u_bg=0.5, gamma=0.05, seed=3)
    assert mean_stall(cfg, "predictive", 0.6) == mean_stall(cfg, "predictive", 0.6)


@pytest.mark.parametrize("u,gamma", [(0.0, 0.0), (0.0, 0.05), (0.25, 0.0), (0.5, 0.0), (0.5, 0.05)])
def test_closed_form_is_a_fair_approximation_at_moderate_load(u, gamma):
    sim = breakeven_accuracy(KVConfig(u_bg=u, gamma=gamma))
    assert sim is not None and abs(sim - closed_form(u, gamma)) < 0.2


def test_closed_form_fails_at_high_load_because_wasted_prefetches_saturate_the_link():
    # documented limit of the formula: with u_bg = 0.9 even a perfect-looking threshold (< 0.5) does not pay off in the simulation
    sim = breakeven_accuracy(KVConfig(u_bg=0.9, gamma=0.0))
    assert closed_form(0.9, 0.0) < 0.5 and (sim is None or sim > 0.9)


def test_closed_form_properties_match_the_lean_theorems():
    assert closed_form(0.0, 0.0) == 0.0
    assert closed_form(0.3, 0.1) <= closed_form(0.6, 0.1)            # prefetch_threshold_mono_u (gamma <= 1)
    assert closed_form(0.3, 0.3) < 0.5                                 # prefetch_threshold_lt_half: u + 2*gamma < 1
    assert closed_form(0.4, 1.0) == pytest.approx(1.0)                 # threshold reaches 1 exactly when gamma = 1
