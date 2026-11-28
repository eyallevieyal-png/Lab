# Research status (2026-11-28)

The single page that says what can be relied on today. Every line has an evidence class (see `README.md`). Source of truth for individual results: `EXPERIMENT_REGISTRY.json`.

## 1. Trustworthy today

| Result | Class | Experiment | Limit |
|---|---|---|---|
| The corrected simulator has none of the 8 defects found in the first version | validated_simulation | `verify_simulator_fixes` (8/8) | validity of the code, not of the model's assumptions |
| In the corrected simulator the controller trusting its predictor (Predictive) is the worst controller (mean regret about 1.1-1.5 ms vs 0.11 for data-local Reactive); gated controllers stay close to data-local; the oracle has regret 0 | validated_simulation | EXP-SIM-CTRL-001 (60 runs, 5 seeds) | assumed tier/precision catalogue; 2 workloads |
| Single-stream decode is 50-300x memory-bound; prefill is compute-bound from a configuration-specific token count (316 / 300 bf16, 80 int4 weights-only) | theoretical (specs) + reproduced | EXP-H6-CAL-001 | first-order bound; H100 SXM spec page and Llama 3.1 configs only |
| Memory share of time increases with compute speed-up in the additive model | theoretical (Lean, builds) | PROOF-LEAN-001 | inside the model |
| Break-even accuracy `(u+gamma)/(1+u)` is a theorem; it matches a queueing simulation up to u of about 0.5 and breaks above | theoretical + validated_simulation + reproduced | EXP-H13-FORM-001 / ANA-001 | Poisson background, deterministic service |
| MoE crossover ratios 9.9x (Qwen3-30B-A3B) and 3.7x (Mixtral-8x7B) | **theoretical estimate** + reproduced | EXP-H6-MOE-001 | uniform routing, 8-position speculative verification, context 512, bf16. Not measured |
| Laptop CPU roofline: ratio 0.94 at B=1, 0.97-1.08 at B>=256, 3.3-3.7x slower between | preliminary_hardware | EXP-HW-CPU-001 | one machine |
| T4: 276 GB/s, 24.9 fp16 TFLOPS, roofline about 1.4x optimistic when memory-bound, crossover predicted 92 vs measured 64-128, Colab host link about 1.0-2.1 GB/s | preliminary_hardware | EXP-HW-T4-001 | **one run**, one Colab session; not an H100 and not a spec |

## 2. Not trustworthy / archived
- Everything from the first simulator (commit `feca1e6`), including the old `results_summary.json`: `archive/`, **superseded**.
- The "72-run" evaluation (-24.6% P99 etc.): no code or data in this repository; **superseded and unverified**; must not be presented as a current finding.

## 3. Unverified hypotheses
H16 placement, H17 contention-aware prefetch, H18 MoE-aware orchestration, H19 multi-workload, H20 hardware validation: designed in `NEXT_EXPERIMENTS.md`. H16 and H17 have a first simulated pass (below); the others are not run.

## 4. Orchestrator prototype: what it shows (EXP-ORCH-001, simulated)

Prototype with simulated tiers and **assumed** parameters; CXL is a model only. 20 seeds per cell, paired differences on identical traces, 95% CI; ✱ = CI excludes 0 (no multiple-comparison correction). Table: change in **P99 latency versus the LRU baseline** (negative is better).

| Scenario | Background | always_prefetch | static_threshold | adaptive_gate (H13) | adaptive (gate + placement) | clairvoyant bound |
|---|---|---|---|---|---|---|
| agentic_loop (predictable) | idle | -51.0% ✱ | -50.9% ✱ | -43.6% ✱ | -41.3% ✱ | -76.5% |
| agentic_loop | 50% on DRAM link | -69.9% ✱ | -68.3% ✱ | -52.1% ✱ | -54.1% ✱ | -80.5% |
| zipf_iid | idle | +0.1% | -0.2% | -0.9% | **-9.8% ✱** | -67.5% |
| zipf_iid | 50% | +4.9% | 0.0% | 0.0% | **-14.6% ✱** | -68.4% |
| uniform (no structure) | idle | **+17.7% ✱** | 0.0% | 0.0% | +5.5% ✱ | -72.3% |
| uniform | 50% | **+99.2% ✱** | 0.0% | 0.0% | +11.1% ✱ | -71.2% |
| phase_shift | idle | +5.4% ✱ | 0.0% | 0.0% | **+46.5% ✱** | -57.1% |
| phase_shift | 50% | **+77.7% ✱** | 0.0% | 0.0% | +0.8% | -64.2% |

Reading it honestly:
1. **Prediction pays when the workload is predictable** (P99 roughly halved) and **can double the P99 when it is not** (always_prefetch on uniform traffic with a busy link). Any adaptive scheme has to avoid the second case.
2. **H17 (contention-aware gating): only partly supported.** The H13 gate avoids the catastrophic cases, but so does a static "accuracy > 0.5" rule, and on the predictable workload the H13 gate is *significantly worse* than the static rule (P99 38.6 vs 25.5 ms under background load; 16.9 vs 14.7 idle). Diagnosis on separate seeds (100-119, agentic_loop, 50% background): P99 minus always-prefetch = +16.7 ± 3.9 ms (measured u), +10.9 ± 3.9 (true u), +9.4 ± 2.3 (u ignored), +12.4 ± 3.8 (no queue check), +13.5 ± 3.0 (no margin); static threshold -0.4 ± 1.0. **So the estimate of the background load is not the main cause; the threshold model itself is conservative here.** The mechanism is not isolated; hypotheses are in `NEXT_EXPERIMENTS.md`.
3. **H16 (placement): not supported as a general claim.** Reuse-score eviction and demotion helps on Zipf traffic (-9.8% / -14.6%) and hurts on a phase-shifting working set (+46.5%, idle link) and slightly on uniform traffic. Hypothesis for the regression (untested): the score decays too slowly after a working-set change.
4. **The clairvoyant bound is far below every realisable policy** (-57% to -80%): predictor quality, not the gate, is the dominant limit.
5. A first version of the gate was fed total link utilisation (own traffic included), which double-counts it. Single-seed check of the fix (seed 1, agentic_loop, `adaptive_gate`): idle link mean latency 9.7 -> 7.5 ms and P99 21.5 -> 20.8 ms; with 50% background mean 17.9 -> 11.1 ms and P99 56.6 -> 55.4 ms. The reported table is the corrected version. The fix did not change the conclusion that the gate trails the static threshold.

## 5. What can be demonstrated today
- A reproducible control-loop prototype with simulated tiers: `python -X utf8 -m orchestrator.bench` and the dashboard `research/figures/orchestrator_dashboard.html`.
- That the simulator is free of the 8 known defects, in code.
- Roofline-based decode/prefill reasoning from public specs, and one T4 sanity check.

## 6. What needs access we do not have
- **H100-class GPU** (not free): serving benchmark, real prefill/decode/TTFT/P99, energy, interference. Plan: `HARDWARE_VALIDATION_PLAN.md`.
- **CXL-capable server:** any claim about CXL placement. Until then CXL is a modelled tier.
- **Real KV-reuse / MoE routing traces:** H18 and a fair test of H16/H17 on real workloads.
- **A free Colab/Kaggle session by the owner:** T4 repeats (`experiments/run_gpu_repeats.py`), the cheapest way to strengthen the only hardware evidence.

## 7. New research opportunities that emerged
1. Why does a theorem-derived gate lose to a trivial threshold? (H17 diagnosis; possibly a publishable negative result about using closed-form thresholds online.)
2. Predictor quality is the bottleneck (clairvoyant bound vs realisable): learned or semantic predictors for agentic/tool-call sessions.
3. Placement policies need change detection (phase shifts); adaptive decay.
4. The gap between the T4 host link in Colab (about 1-2 GB/s) and the assumed PCIe Gen5 (64 GB/s): link bandwidth dominates every conclusion about prefetch; measuring it on real servers is the highest-leverage hardware fact.
