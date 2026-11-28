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
H16 placement and H17 contention-aware prefetch have simulated passes (section 4); H18 MoE-aware orchestration, H19 multi-workload and H20 hardware validation are designed in `NEXT_EXPERIMENTS.md` and not run.

## 4. Orchestrator prototype: what it shows (EXP-ORCH-001, simulated)

Prototype with simulated tiers and **assumed** parameters; CXL is a model only. 20 seeds per cell (0-19), paired differences on identical traces, 95% CI; ✱ = CI excludes 0 (no multiple-comparison correction). Cells: **P99 latency versus the LRU baseline** (negative is better). **This table is generated from `research/results/orchestrator_bench.json`, not typed by hand** (an earlier hand-typed version of it carried one significance mark the data does not support).

| Scenario | Background | always | static 0.5 | gate (H13, v1) | gate v2 | adaptive (v1) | adaptive v2 | clairvoyant bound |
|---|---|---|---|---|---|---|---|---|
| agentic_loop | idle link | -51.0% ✱ | -50.9% ✱ | -43.6% ✱ | -50.8% ✱ | -41.3% ✱ | -51.5% ✱ | -76.5% ✱ |
| agentic_loop | 50% on DRAM link | -69.9% ✱ | -68.3% ✱ | -52.1% ✱ | -58.6% ✱ | -54.1% ✱ | -60.0% ✱ | -80.5% ✱ |
| zipf_iid | idle link | +0.1% | -0.2% | -0.9% | -2.5% | -9.8% ✱ | -12.2% ✱ | -67.5% ✱ |
| zipf_iid | 50% on DRAM link | +4.9% | +0.0% | +0.0% | -5.3% ✱ | -14.6% ✱ | -15.9% ✱ | -68.4% ✱ |
| uniform | idle link | +17.7% ✱ | +0.0% | +0.0% | +0.0% | +5.5% | +5.0% ✱ | -72.3% ✱ |
| uniform | 50% on DRAM link | +99.2% ✱ | +0.0% | +0.0% | +0.0% | +11.1% ✱ | +10.6% ✱ | -71.2% ✱ |
| phase_shift | idle link | +5.4% ✱ | +0.0% | +0.0% | +0.0% | +46.5% ✱ | -0.2% | -57.1% ✱ |
| phase_shift | 50% on DRAM link | +77.7% ✱ | +0.0% | +0.0% | +0.0% | +0.8% | +10.1% ✱ | -64.2% ✱ |

`gate v2` and `adaptive v2` were designed AFTER diagnoses run on other seeds (100-119), so seeds 0-19 are held out for them; `gate (v1)` and `adaptive (v1)` are the pre-diagnosis versions, kept for the record. The stored JSON now also holds every per-seed row (`raw_rows`).

Reading it honestly:
1. **Prediction pays when the workload is predictable** (P99 roughly halved, or cut by two thirds under background load) and **can double it when it is not** (always-prefetch: +99% on unstructured traffic with a busy link, +78% on the shifting working set). Any adaptive scheme has to avoid the second case; a static accuracy threshold of 0.5 already does.
2. **H17, contention-aware gating: partly supported, in a narrow regime.**
   - The H13 gate as first implemented trailed the static threshold on predictable traffic (-43.6% / -52.1% versus -50.9% / -68.3%). Diagnosis on seeds 100-119 (EXP-H17-DIAG-001/002) found three separable causes: (a) its threshold is computed in units of the *source tier's* transfer time, so it jumps between about 0.22 and 0.64 depending on where the missing blocks sit (this alone explains the idle-link gap and about a third of the loaded gap); (b) a queue-length veto that skips a prefetch exactly when a miss would cost most; (c) the measured background-load estimate (with (a) and (b) fixed, P99 minus always-prefetch is +8.1 ms with measured load, +1.9 ± 2.0 with the true load, -0.5 ± 0.5 with load ignored).
   - `gate v2` removes (a) and (b). It ties the static rule on an idle link (-50.8% vs -50.9%) but still trails it with 50% background on accurate predictions (-58.6% vs -68.3%).
   - **Where load-awareness earns its keep** (EXP-H17-SWEEP-001/002, held-out seeds 0-19; predictor accuracy controlled at 0.3 / 0.5 / 0.7): with medium accuracy (0.3 and 0.5) and heavy background load (u = 0.75) the v2 gate beats the best static threshold chosen in hindsight by 373 ± 74 ms and 523 ± 158 ms of P99 (significant); at u = 0.5 it ties; with accurate predictions (0.7) at u = 0.5 it loses to a static threshold by 9.6 ± 3.5 ms. **Caveat:** at u = 0.75 the link is overloaded and absolute P99 values are around a second and depend on the run length; only the relative comparison is meaningful.
3. **H16, placement: mixed; one regression understood, one new one introduced.**
   - The first rule set (reuse-score eviction plus "cold data to the slower tier first") regressed +46.5% on the shifting working set. Ablation (seeds 100-119): the demotion rule causes it (demote-only +3.5 ms, evict-only -0.1 ms); slow score decay makes it worse (0.85: +3.6 ms, 0.97: +6.0 ms, 0.99: +14.9 ms), and the combination is worse than either part. Faster decay alone leaves a significant residue and costs the Zipf gain.
   - `adaptive v2` (score eviction only, ordinary waterfall demotion) removes the idle-link regression (-0.2%) and improves Zipf (-12.2% idle, -15.9% loaded), but is **still worse on unstructured traffic (+5.0% / +10.6%) and on the shifting working set under background load (+10.1%)**. Not a general improvement; hypotheses for the remaining regressions are untested.
4. **The clairvoyant bound is far below every realisable policy** (-57% to -80%): predictor quality, not the gate, is the dominant limit.
5. History: the first version of the gate was fed total link utilisation (own traffic included), double-counting it; fixed before the numbers above. A second engine bug (the model's own prediction probability was reset to 0 in the second context) affected no reported number, because no policy used that field before the diagnosis.

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
1. A theorem-derived gate lost to a trivial threshold for three identifiable reasons, and wins only in a narrow regime (medium accuracy, heavy load): a concrete, checkable lesson about using closed-form thresholds online. Worth testing on real traces before saying anything outside this simulator.
2. Predictor quality is the bottleneck (clairvoyant bound vs realisable): learned or semantic predictors for agentic/tool-call sessions.
3. Placement policies need change detection (phase shifts); adaptive decay.
4. The gap between the T4 host link in Colab (about 1-2 GB/s) and the assumed PCIe Gen5 (64 GB/s): link bandwidth dominates every conclusion about prefetch; measuring it on real servers is the highest-leverage hardware fact.
