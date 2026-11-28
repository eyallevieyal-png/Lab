# Next experiments (H16-H20), designed before running

Rule: no large experiment starts until its design below is reviewed and the cheap validation in the last column has passed. Every experiment gets an ID in `EXPERIMENT_REGISTRY.json` and is labelled simulated / measured. Simulation and hardware are never pooled.

**What the prototype already says about H16 and H17** (EXP-ORCH-001, simulated, 20 seeds, details in `RESEARCH_STATUS.md`): H17 is only partly supported and H16 is not supported as a general claim. These results sharpen the designs below; they do not replace the hardware tests.

## Priority order

| Rank | ID | Why this rank | Cost | Blocker |
|---|---|---|---|---|
| 1 | H20a (T4 repeats) | every other claim leans on one T4 run; free; tooling is ready | free, ~1 h of the owner's time | owner runs a notebook |
| 2 | H17 (contention-aware prefetch) | the prototype shows the H13 gate is *worse* than a static threshold on the predictable workload: the central open question and the likely independent contribution | free (CPU simulation) | none |
| 3 | H16 (placement) | mixed result, one hypothesised cause (stale scores) is cheap to test | free | none |
| 4 | H18 (MoE-aware) | needs real routing traces; MoE results so far are computed, not measured | free if public traces exist | find a routing trace |
| 5 | H19 (multi-workload) | depends on H16/H17 being settled first | free | H16/H17 |
| 6 | H20b (H100) | the decisive validation | **not free** | access decision by the owner |

## H16: adaptive memory placement
- **Hypothesis:** placing and evicting by predicted reuse and transfer cost lowers latency versus an LRU waterfall.
- **Baseline:** `lru_demand`. **Also compare:** LFU, ARC-style, and `adaptive` with each component ablated (eviction only, demotion only).
- **Independent variables:** workload (agentic_loop, zipf, uniform, phase_shift, plus a trace with measured reuse), score decay constant (0.9-0.995), HBM/DRAM capacity ratio, tier bandwidth ratios.
- **Dependent:** mean and P99 latency, HBM block hit rate, bytes moved. **Control:** arrivals, trace, predictor, tier parameters (common random numbers per seed).
- **Repetitions:** 30 seeds per cell. **Statistics:** paired difference vs baseline, 95% CI (t), Holm correction across cells.
- **Success:** paired CI excludes 0 in the favourable direction on at least the workloads with reuse structure AND no significant degradation elsewhere. **The prototype already violates the second clause on `phase_shift` (+46% P99, idle link).**
- **Hypothesised cause to test (not yet tested):** score decay too slow after a working-set change. Test: sweep the decay constant and add change detection. A failure to fix it is a result.
- **Confounders:** the `phase_shift` and `agentic_loop` generators are mine; a trace from a real serving system may behave differently. Free demotion writes favour all demotion-heavy policies.
- **Cheap validation first:** unit tests already cover capacity and conservation; add a test that a policy equal to LRU reproduces `lru_demand` exactly.

## H17: contention-aware predictive prefetching
- **Hypothesis:** queue-aware prediction reduces P99 latency without increasing total bandwidth consumption.
- **Baselines:** `lru_demand`, `always_prefetch`, **`static_threshold`** (accuracy > 0.5), `adaptive_gate`, clairvoyant bound.
- **Independent:** background load u in {0, 0.25, 0.5, 0.75}, predictor accuracy (controlled by noise injected into a known successor structure, plus the real Markov predictor), lead time, burstiness (Poisson vs a recorded trace such as BurstGPT arrival gaps).
- **Dependent:** P99, mean, throughput, bytes moved, wasted prefetch bytes, queue length, link utilisation, stability (does queue length stay bounded as load rises?).
- **Prototype finding to explain:** under u = 0.5 on a predictable workload (20-seed means) the gate ended at P99 38.6 ms versus 24.3 ms for always-prefetch and 80.6 ms for no prefetch. In a single-seed check it skipped about 30% of the candidate prefetches (948 issued versus 1347 for always-prefetch); the per-seed prefetch counts are not stored in the benchmark JSON yet and should be. The static threshold matched the gate's protection on unpredictable workloads and beat it on the predictable one.
- **Diagnosis step 1 DONE (EXP-H17-DIAG-001, seeds 100-119):** with the true u the gate is still +10.9 ± 3.9 ms P99 worse than always-prefetch, with u ignored +9.4 ± 2.3, without the queue check +12.4 ± 3.8; the static threshold is -0.4 ± 1.0. So estimation error is not the main cause. **Remaining hypotheses (untested):** the M/D/1 threshold assumes independent epochs and Poisson background, while here the request stream is correlated and the lead equals the period; the running accuracy hovers near the threshold, so small fluctuations switch prefetching off. Next tests: compare analytic and engine stall for identical parameters; sweep the accuracy estimator window.
- **Repetitions:** 30 seeds. **Statistics:** paired CIs; the claim "gate beats static threshold" needs a CI excluding 0 on the predictable workload, which is currently false (it is significantly worse).
- **Success:** P99 not worse than the best static rule in any cell, and strictly better in at least one cell with a bursty background. Otherwise report that the closed-form threshold is not useful as an online gate.
- **Confounders:** utilisation estimator window, prefetch lead, FIFO channel assumption (real links are not strictly FIFO).
- **Cheap validation first:** the stall-model comparison above.

## H18: MoE-aware memory orchestration
- **Hypothesis:** using expert reuse patterns for residency decisions reduces expert-weight movement at equal or better latency.
- **Baseline:** LRU over experts; static top-k pinned experts. **Independent:** routing skew (uniform to heavy-tailed), number of experts resident, link bandwidth, batch size, speculative verification width.
- **Dependent:** expert hit rate, bytes moved, step latency. **Control:** model dimensions (Qwen3-30B-A3B and Mixtral-8x7B configs), traces.
- **Workload:** real routing traces from a public MoE inference run if available; otherwise synthetic skew, **labelled synthetic**. The existing 9.9x and 3.7x crossover ratios are theoretical estimates under uniform routing and must not be used as evidence for this experiment.
- **Repetitions:** 30 seeds for synthetic, all available traces for real. **Statistics:** as H16.
- **Success:** a significant reduction in bytes moved on skewed routing without a latency regression on uniform routing.
- **Confounders:** routing is workload- and model-specific; expert-parallel communication is out of scope.
- **Cheap validation first:** find out whether any public routing trace exists. If none does, the experiment stays synthetic and is reported that way.

## H19: multi-workload orchestration
- **Hypothesis:** a global controller improves aggregate and per-tenant tail latency when several inference workloads compete for the same memory and link.
- **Baseline:** per-workload independent controllers (each with a private share), and a shared LRU. **Independent:** number of tenants (2-8), workload mix, arrival correlation, fairness weights.
- **Dependent:** per-tenant P99, aggregate throughput, Jain fairness index, link utilisation. **Control:** total HBM and link capacity.
- **Repetitions:** 30 seeds. **Statistics:** paired CIs per tenant; report the worst tenant, not only the mean.
- **Success:** better aggregate P99 without making any tenant significantly worse.
- **Confounders:** tenant start offsets, the prototype's single-server assumption (must be extended to concurrent servers first).
- **Cheap validation first:** extend the engine to several servers and test that one tenant reproduces the single-tenant results exactly.

## H20: real hardware validation
- **Hypothesis:** the simulator's predicted behaviour (roofline crossover, memory-bound ratio, H13 break-even) reproduces on real GPU hardware.
- **Stages and criteria:** `HARDWARE_VALIDATION_PLAN.md` (A: T4 repeats; B: second card; C: H100-class, not free).
- **Success criteria are pre-registered there** (crossover within a factor 2; ratio inside 1.0-1.5; H13 break-even within 0.1; orchestrator benefit only if the paired CI excludes 0).
- **Confounders:** virtualised hosts (the Colab host link is about 30-60x slower than a PCIe Gen5 assumption), clock throttling, neighbours.

## Statistical rules for all of the above
Multiple seeds (at least 20, 30 for new claims); common random numbers across policies; paired differences with t-based 95% CIs; Holm or Bonferroni correction when several cells are tested together (the current benchmark does not correct, which is stated on its dashboard); effect sizes next to p-values; no tuning on the evaluation seeds (tune on seeds 100-119, evaluate on 0-19); report null and negative results.
