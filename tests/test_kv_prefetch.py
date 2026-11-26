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


def test_md1_wait_and_stability_floor():
    from src.kv_prefetch import md1_wait, stability_floor
    assert md1_wait(0.5) == pytest.approx(0.5) and md1_wait(1.0) == float("inf")
    u, n = 0.75, 6.0
    floor = stability_floor(u, n)
    assert u + (2 - (floor + 0.01)) / n < 1 < u + (2 - (floor - 0.01)) / n          # Lean: stable_iff


def test_analytic_threshold_is_none_when_reactive_already_saturates_and_tracks_the_simulation_otherwise():
    from src.kv_prefetch import analytic_breakeven
    assert analytic_breakeven(0.9, 0.0, 1.5, 6.0) is None                            # u + 1/period >= 1 (Lean: no_stable_predictor)
    sim = breakeven_accuracy(KVConfig(u_bg=0.75, gamma=0.0, period=12.0))
    ana = analytic_breakeven(0.75, 0.0, 1.5, 12.0)
    assert abs(sim - ana) < 0.1 and abs(sim - closed_form(0.75, 0.0)) > 0.1          # the plain formula misses here, the congestion-aware one does not


def test_online_policy_is_not_worse_than_the_best_fixed_policy_on_poisson_traffic():
    for u, p in ((0.5, 0.3), (0.5, 0.9), (0.65, 0.6)):
        cfg = KVConfig(u_bg=u, gamma=0.05, lead=1.5, period=12.0, epochs=2000)
        r, a, o = (mean_stall(cfg, pol, p) for pol in ("reactive", "predictive", "online"))
        assert o <= min(r, a) * 1.1 + 0.02
