import math

import pytest

from orchestrator.bench import mean_ci, tcrit
from orchestrator.dashboard import write_dashboard
from orchestrator.engine import Config, Engine, run_one
from orchestrator.estimator import TransferCostEstimator
from orchestrator.policies import ALL_POLICIES, AdaptiveGate, AdaptiveGateV2, AdaptiveV2
from orchestrator.telemetry import Context, MarkovPredictor
from orchestrator.tiers import DEFAULT_TIERS, SimulatedBackend
from orchestrator.workload import SCENARIOS, make_trace

SMALL = dict(requests=300, warmup=20)


def test_transfer_time_matches_bandwidth_and_latency():
    b = SimulatedBackend()
    # 1 GB over the 64 GB/s DRAM link = 15.625 ms plus 0.001 ms latency
    assert b.transfer_ms("DRAM", 10**9) == pytest.approx(15.625 + 0.001)


def test_fifo_channel_queues_and_reports_queue_delay():
    b = SimulatedBackend()
    t1 = b.reserve_read("DRAM", 10**9, now=0.0)
    t2 = b.reserve_read("DRAM", 10**9, now=0.0)
    assert t2 == pytest.approx(t1 + b.transfer_ms("DRAM", 10**9))
    assert b.queue_ms("DRAM", 0.0) == pytest.approx(t2)
    assert b.queue_ms("CXL", 0.0) == 0.0      # other channels are independent


def test_foreign_utilization_excludes_own_transfers_and_sees_background():
    own = SimulatedBackend()
    own.reserve_read("DRAM", 4 * 10**9, now=0.0)
    assert own.utilization("DRAM", 100.0) > 0.0
    assert own.utilization("DRAM", 100.0, foreign_only=True) == 0.0
    bg = SimulatedBackend(bg_util={"DRAM": 0.5}, seed=3)
    assert 0.2 < bg.utilization("DRAM", 5000.0, foreign_only=True) < 0.9


def test_predictor_is_causal_and_learns_a_cycle():
    p = MarkovPredictor()
    seq = [0, 1, 2, 3] * 30
    for s in seq:
        p.observe(s)
        p.make_prediction(s)
    assert p.accuracy > 0.85
    # accuracy is scored only against a prediction made BEFORE the outcome: a fresh predictor has none
    q = MarkovPredictor()
    q.observe(5)
    assert q.accuracy == 0.0 and q.n == 0


def test_only_clairvoyant_policies_receive_the_future():
    seen = {}

    class Spy(AdaptiveGate):
        def prefetch(self, ctx, missing, est):
            seen.setdefault("spy", []).append(ctx.future_hint)
            return None

    for cls, name in ((Spy, "spy"), (ALL_POLICIES["oracle_prefetch"], "oracle")):
        Engine(Config(**SMALL), cls()).run()
    assert set(seen["spy"]) == {None}
    assert ALL_POLICIES["oracle_prefetch"].clairvoyant and not Spy.clairvoyant


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("policy", list(ALL_POLICIES))
def test_capacity_and_conservation_invariants(scenario, policy):
    e = Engine(Config(scenario=scenario, seed=2, bg_util={"DRAM": 0.3}, **SMALL), ALL_POLICIES[policy]())
    e.run()
    for t in DEFAULT_TIERS:
        if t.capacity_blocks is not None:
            assert len(e.lru[t.name]) <= t.capacity_blocks, f"{t.name} over capacity"
    placed = [b for t in e.lru for b in e.lru[t]]
    assert len(placed) == len(set(placed)) == len(e.where)                     # every block lives in exactly one tier
    assert all(b in e.lru[e.where[b]] for b in e.where)                         # the location table agrees with the tier contents


def test_deterministic_and_common_random_numbers():
    a = run_one(Config(seed=4, **SMALL), "adaptive")
    b = run_one(Config(seed=4, **SMALL), "adaptive")
    assert a == b
    assert make_trace("zipf_iid", 48, 100, 1) == make_trace("zipf_iid", 48, 100, 1)
    assert make_trace("zipf_iid", 48, 100, 1) != make_trace("zipf_iid", 48, 100, 2)


def test_oracle_bound_is_at_least_as_good_as_the_baseline():
    base = run_one(Config(scenario="agentic_loop", seed=1, **SMALL), "lru_demand")
    orc = run_one(Config(scenario="agentic_loop", seed=1, **SMALL), "oracle_prefetch")
    assert orc["mean_latency_ms"] < base["mean_latency_ms"]


def test_gate_refuses_hopeless_prefetching_that_always_prefetch_wastes():
    cfg = dict(scenario="uniform", seed=1, bg_util={"DRAM": 0.5}, requests=600, warmup=50)
    always = run_one(Config(**cfg), "always_prefetch")
    gate = run_one(Config(**cfg), "adaptive_gate")
    assert gate["prefetches_issued"] < 0.05 * always["prefetches_issued"]
    assert gate["wasted_prefetch_gib"] < 0.05 * always["wasted_prefetch_gib"]
    assert gate["p99_ms"] < always["p99_ms"]


def test_gate_prefetches_when_the_predictor_is_accurate():
    r = run_one(Config(scenario="agentic_loop", seed=1, **SMALL), "adaptive_gate")
    assert r["prefetches_issued"] > 50 and r["predictor_accuracy"] > 0.6


def test_confidence_interval_helpers():
    m, ci = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0
    assert ci == pytest.approx(tcrit(4) * math.sqrt(2.5) / math.sqrt(5))
    assert tcrit(9) == pytest.approx(2.262) and tcrit(100) == 1.96
    assert math.isnan(mean_ci([1.0])[1])


def test_dashboard_labels_simulation_and_is_self_contained(tmp_path):
    pol = {"lru_demand": {"p99_ms": {"mean": 10, "ci95": 1}, "mean_latency_ms": {"mean": 5, "ci95": 0.5}, "hbm_block_hit_rate": {"mean": 0.5}, "wasted_prefetch_gib": {"mean": 0}},
           "oracle_prefetch": {"p99_ms": {"mean": 5, "ci95": 1}, "mean_latency_ms": {"mean": 3, "ci95": 0.5}, "hbm_block_hit_rate": {"mean": 1}, "wasted_prefetch_gib": {"mean": 0}}}
    payload = {"metadata": {"seeds": [0], "baseline": "lru_demand", "generated_utc": "t", "repo_head": "h", "regenerate": "x", "scenarios": {"s": "d"}},
               "cells": [{"scenario": "s", "background": "idle link", "policies": pol}]}
    out = tmp_path / "d.html"
    write_dashboard(payload, str(out))
    text = out.read_text(encoding="utf-8")
    assert "Simulated, not measured" in text and "<script" not in text and "http" not in text.replace("https://", "")


def test_v2_gate_decision_does_not_depend_on_the_source_tier_but_the_original_does():
    est = TransferCostEstimator(SimulatedBackend())
    blk = 32 * 2 ** 20
    mk = lambda acc: Context(now=0.0, session=1, util={"DRAM": 0.2, "CXL": 0.2, "NVMe": 0.2}, queue_ms={"DRAM": 0.0, "CXL": 0.0, "NVMe": 0.0}, pred=2, pred_p_model=0.8,  # noqa: E731
                             pred_accuracy=acc, mean_gap_ms=14.0, access_score={}, extra={"session_blocks": 8, "block_bytes": blk})
    from_dram, from_cxl = {2: {"DRAM": 8 * blk}}, {2: {"CXL": 8 * blk}}
    v2, orig = AdaptiveGateV2(), AdaptiveGate()
    accs = [a / 100 for a in range(10, 100, 5)]
    assert all((v2.prefetch(mk(a), from_dram, est) is not None) == (v2.prefetch(mk(a), from_cxl, est) is not None) for a in accs)
    assert any((orig.prefetch(mk(a), from_dram, est) is not None) != (orig.prefetch(mk(a), from_cxl, est) is not None) for a in accs)   # the defect v2 removes


def test_v2_has_no_queue_veto_but_the_original_has():
    est = TransferCostEstimator(SimulatedBackend())
    blk = 32 * 2 ** 20
    ctx = Context(now=0.0, session=1, util={"DRAM": 0.0, "CXL": 0.0, "NVMe": 0.0}, queue_ms={"DRAM": 100.0, "CXL": 0.0, "NVMe": 0.0}, pred=2, pred_p_model=0.9,
                  pred_accuracy=0.9, mean_gap_ms=14.0, access_score={}, extra={"session_blocks": 8, "block_bytes": blk})
    need = {2: {"DRAM": 8 * blk}}
    assert AdaptiveGate().prefetch(ctx, need, est) is None and AdaptiveGateV2().prefetch(ctx, need, est) == 2


def test_agentic_accuracy_scenario_variant_controls_predictability():
    hi = run_one(Config(scenario="agentic_loop@0.9", seed=1, **SMALL), "lru_demand")["predictor_accuracy"]
    lo = run_one(Config(scenario="agentic_loop@0.3", seed=1, **SMALL), "lru_demand")["predictor_accuracy"]
    assert hi > lo + 0.3


def test_adaptive_v2_evicts_by_score_but_demotes_like_the_baseline():
    ctx = Context(now=0.0, session=1, util={}, queue_ms={}, pred=None, pred_p_model=0.0, pred_accuracy=0.0, mean_gap_ms=14.0, access_score={1: 5.0, 2: 0.1}, extra={})
    v2 = AdaptiveV2()
    assert v2.hbm_victim([(1, 0), (2, 0)], ctx) == (2, 0)                      # lowest reuse score goes first, not the oldest
    assert ALL_POLICIES["lru_demand"]().hbm_victim([(1, 0), (2, 0)], ctx) == (1, 0)
    free = {"DRAM": 0, "CXL": 5, "NVMe": 10**9}
    assert v2.demote_tier((2, 0), ctx, free) == "CXL" == ALL_POLICIES["lru_demand"]().demote_tier((2, 0), ctx, free)   # waterfall: first tier with room
    assert v2.lower_victim("DRAM", [(7, 0), (8, 0)], ctx) == (7, 0)             # LRU in the lower tiers
