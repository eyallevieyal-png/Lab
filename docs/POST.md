# Auditing an LLM-inference simulator: a roofline, a prefetch threshold, MoE, and 20 Lean theorems (with the limits stated)

*Everything below is simulation, calculation from public specifications, or proof inside a stated model, unless marked "measured". The only measurement is on a laptop CPU. There is no GPU measurement yet, and that is the main limitation. Code, tests and Lean sources: this repository, branch `fix/simulator-validity` (merged to `main`). Full assumptions: [MODEL.md](MODEL.md).*

## TL;DR

- A research simulator about where data should live during LLM inference had 8 defects that made its headline results artifacts (for example, the "reactive" baseline returned the oracle's answer). After fixing them, some conclusions flipped.
- From public H100 specs and Llama 3.1 configs: single-stream decode is memory-bound by 50-300x, and prefill becomes compute-bound at about 300 tokens. "Memory dominates as compute gets cheaper" holds for decode and fails for prefill, so it has to be stated per phase.
- For prefetching an offloaded KV cache there is a closed-form accuracy threshold `p* = (u + γ) / (1 + u)` (proved in Lean 4). It matches a queueing simulation up to link utilisation of about 0.5 and breaks above that, for a reason I could identify and test.
- For mixture-of-experts, the batch size at which decode becomes compute-bound is about (total / active parameters) times that of a dense twin: 9.9x for Qwen3-30B-A3B and 3.7x for Mixtral-8x7B under 8-position speculative verification (computed, assuming uniform routing).
- This is not a breakthrough. Several conclusions are known in the literature (section 8). What I would stand behind is the method: audit the baseline, separate proved / simulated / computed-from-specs / measured / assumed, and publish the limits.

## 1. How each claim is labelled

| Label | Meaning here |
|---|---|
| **Proved** | Lean 4 + Mathlib theorem, compiles, no `sorry`, standard axioms only. True inside the stated model. |
| **Simulated** | Output of the simulator. Depends on my assumptions, not on hardware. |
| **Computed from specs** | First-order roofline from vendor numbers (H100) and public model configs. An ideal bound, not a measurement. |
| **Measured** | Actually timed. Only one: a laptop CPU, not a GPU. |
| **Assumed** | A chosen parameter (data size, link utilisation, routing). |

## 2. The simulator was wrong

I started from a simulator of data placement across HBM, DRAM, CXL, NVMe and a QPU, with several controllers (reactive, predictive, adaptive, oracle). Reading the code and re-running it showed:

| # | Defect | Effect |
|---|---|---|
| 1 | Workloads seeded with Python's per-process `hash()` | Two runs of identical code differed (72 of 299 fields) |
| 2 | "Reactive" returned the oracle's resource | "Is prediction worth its cost?" was "never" by construction |
| 3 | "Regret" was a fixed fraction of cost | Even the oracle had positive regret, sometimes more than Reactive |
| 4 | Memory cost computed, never used | The memory side of the main hypothesis could not affect any result |
| 5 | Prediction cost never charged | Predictors looked free |
| 6 | Prediction correct with probability equal to its confidence | Calibration error could not occur |
| 7 | `seed` did not change the workload | Different seeds gave identical results |
| 8 | Bandwidth divided by 8; data so small transfers cost about zero | Transfers 8x too slow, then irrelevant |

Fixed in 8 changes with tests (61 at the time, 73 now). In the corrected model, prediction does not beat data-local placement unless gated, and gating only makes it match. **Earlier results from the first version should be discarded.**

## 3. H6: does memory take a growing share as compute gets cheaper?

In an additive cost model this is a theorem: memory share `M / (C/k + M)` is strictly increasing in the compute speed-up k (`memShare_strictMono`), and the simulator matches the closed form exactly. The empirical question is where the crossover `k* = C/M` lies, and that is set by data size and bandwidth, which are assumptions. In the simulator, only HBM and the QPU are compute-dominated at baseline; DRAM, CXL and NVMe are memory-dominated at every size tested. In its precision catalog, int4 shrinks bytes more than compute, the opposite of H6's premise (catalog values are unvalidated).

**Calibrated on public specs** (NVIDIA H100 page; Llama 3.1 8B/70B `config.json`; dense peak = half the "with sparsity" figure, which is my derivation):

- Batch-1 decode: compute time is 0.3-2% of memory time at every context from 512 to 32768.
- Often there is no balance point at all: the KV cache is read per sequence, so attention keeps the step memory-bound at any batch size. On H100 SXM, Llama 3.1 8B has none at any tested context.
- bf16 to fp8 moves the balance point by exactly 1.00x (compute peak doubles, bytes halve). Weight-only int4 does the opposite of H6 (less memory-bound).
- Prefill: compute-bound from about 300 prompt tokens (S* = 316 for 8B, 300 for 70B); at 2048 tokens compute time is 6.6-6.9x memory time.

So H6 is true in one phase and false in the other. The critical values are a closed form of a known conclusion.

**Measured (laptop CPU i5-1035G1, float32, numpy/OpenBLAS; not a GPU).** Bandwidth about 13.7 GB/s, matmul about 222 GFLOPS, ridge about 16 FLOP/byte. `Y = W @ X` with W 8192x8192:

- B = 1: measured/predicted 0.94. B >= 256: 0.97-1.08. The roofline is accurate at both ends.
- B = 2-32: measured is 3.3-3.7x slower than predicted. A plausible but unverified cause: the BLAS matrix-matrix kernel repacks W.
- Predicted balance batch 33; first measured batch at 50% of peak: 64. So the roofline is optimistic by about 2x exactly where the crossover is.

A 2026 study reports H100 at batch 1 reaching about 27% of its analytic memory floor ([arXiv 2605.30571](https://arxiv.org/pdf/2605.30571)): same direction.

## 4. H13: when is prefetching an offloaded KV cache worth it?

Model: a correct prefetch saves a stall T; a wrong one costs `u·T` (u = link utilisation); the predictor costs `γ·T`. Prediction pays exactly when accuracy exceeds

`p* = (u + γ) / (1 + u)`

**Proved** in Lean (`prefetch_threshold_iff`, `prefetch_threshold_mono_u`, `prefetch_threshold_lt_half`, `prefetch_threshold_lt_one_iff`).

Specs-based timing (H100, bf16 KV of one Llama 3.1 8B sequence over PCIe Gen5; 64 GB/s one-way is my reading of the printed 128 GB/s): fetching takes 0.22 / 1.7 / 11 decode steps at context 512 / 4096 / 32768.

**Simulated** (one FIFO link, Poisson background traffic, 6000 epochs, 3 seeds, common random numbers):

| u (background) | formula p* | simulated p*, lead 1.5 | simulated p*, lead 3 |
|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 |
| 0.25 | 0.200 | 0.144 | 0.033 |
| 0.50 | 0.333 | 0.389 | 0.263 |
| 0.75 | 0.429 | 0.900 | 0.845 |
| 0.90 | 0.474 | 0.999 | 0.999 |

The formula is fair up to u about 0.5 and fails at high load. Cause (checked): wasted prefetches add link load, and once total load reaches 1 the queue grows without bound, so no accuracy below about 0.9 pays off. The theorem is right; what breaks is the assumption that benefit and loss are constants. Using the stream's own load in `u` does not repair the gap (hypothesis tested and rejected).

### Congestion-aware threshold
Approximating the link as M/D/1 with load `rho = u + (2 - p)/period` (including wasted prefetches): stability needs `p > 2 - period·(1 - u)` (proved: `stable_iff`), and if reactive fetching alone saturates the link, no predictor keeps it stable (`no_stable_predictor`). Over 84 grid cells (20 excluded as saturated), mean absolute error of the break-even accuracy is **0.053 (max 0.178) versus 0.176 (max 0.518) for the plain formula**. Where reactive load is below 0.7 the plain formula is slightly better (0.060 vs 0.082). Both model and simulation assume Poisson traffic and deterministic service, so this validates the analysis, not a real system.

### Real bursty arrivals
Using BurstGPT (1.40 million requests over 61 days; arrival timestamps only, no KV-reuse or session data). Inter-arrival CV² is 174 (Poisson: 1). Three 30,000-request windows, rescaled so mean load equals u:

| window | span | per-minute count CV | break-even accuracy at u = 0.5 / 0.65 / 0.75 |
|---|---|---|---|
| 1 | 2.8 h | 0.16 | 0.290 / 0.393 / 0.625 (Poisson: 0.281 / 0.425 / 0.607) |
| 3 | 14.7 h | 1.37 | 0.979 / 0.991 / 0.994 |
| 2 | 67.9 h | 2.67 | 0.999 / 0.999 / 0.999 |

An average-load threshold is valid only for roughly stationary traffic. The non-stationary windows are overloaded regimes created by the rescaling; a real system would be provisioned for its peaks, so they say little about a well-provisioned deployment.

### Online controller
A controller that estimates load from the last 30 transfers and prefetches only above the M/D/1 break-even matches the best fixed policy on Poisson and smooth traffic (within 0.07 transfer times). In the bursty windows it does **not** help (up to about 15% worse than reactive in some cells): the bottleneck there is capacity, not the prefetch decision.

## 5. Mixture-of-experts (computed from specs, nothing measured)

A step of n = batch × spec tokens reads the non-expert weights once and `U(n) = E(1 - (1 - k/E)^n)` experts per layer, assuming independent uniform routing (an assumption). Compute is n × active parameters. The control is a dense twin with the same active parameters and all experts read. Architectures are from public `config.json` files; my parameter counts match published sizes (30.53B total / 3.04B active for Qwen3-30B-A3B; 46.70B / 12.75B for Mixtral-8x7B).

- **Union fraction:** Qwen3-30B-A3B reads 6% of experts at n = 1, 40% at n = 8 (matches the Hugging Face blog), 87% at 32, 98% at 64. Mixtral: 25% at n = 1, 90% at n = 8.
- **Balance batch B\*** (H100 SXM, bf16/fp8, context 512). Plain decode has no balance point. With speculative verification of 4 / 8 positions:

| Model | B\* (MoE) | B\* (dense twin) | Ratio |
|---|---|---|---|
| Qwen3-30B-A3B | 1,612 / 482 | 163 / 49 | 9.9x |
| Mixtral-8x7B | 331 / 148 | 90 / 40 | 3.7x |

At context 4096 only Mixtral with 8 positions balances (B\* = 440).
- **Why the ratio is total/active:** at B\* nearly every expert is already read (union above 99.9%), so weight read time is that of the full model while compute is that of the active part: 30.5/3.04 ≈ 10 and 46.7/12.75 ≈ 3.7. Tested within 12%.
- **Skew** (Zipf-like popularity, approximation) lowers the union at n = 8 from 40% to 28% but does not move B\*, because the union is saturated at the balance point.
- **Capacity:** these batches often exceed what fits next to the weights on one H100 (Qwen bf16, context 512: at most 376 sequences; fp8: 1,965; Mixtral bf16 does not fit). The balance point is mostly unreachable on one card in bf16.
- **Not shown:** real routers are correlated across tokens and layers; expert-parallel communication and load imbalance are not modelled. No Lean proof was added for this part.

## 6. The Lean library

20 theorems in `formal/LabFormal/Orchestration.lean` (Lean 4 + Mathlib, no `sorry`):

| Area | Theorems |
|---|---|
| Net benefit of acting on a prediction | `netBenefit_pos_iff`, `netBenefit_at_threshold`, `netBenefit_mono_p`, `netBenefit_anti_c`, `reactive_optimal_below_threshold` |
| H6 memory share | `memShare_strictMono`, `memShare_mem_Ioo`, `memoryBound_iff` |
| Oracle and regret | `regret_nonneg`, `regret_eq_zero_iff`, `oracle_lower_bound`, `totalRegret_ge_overhead` |
| Prefetch threshold | `prefetch_threshold_iff`, `prefetch_threshold_mono_u`, `prefetch_threshold_lt_half`, `prefetch_threshold_lt_one_iff` |
| Decode step and critical context | `no_balance_point`, `compute_bound_iff`, `kv_dominates_iff`, `critical_context_anti` |
| Queue stability | `stable_iff`, `no_stable_predictor` |

These are proofs about the model, not about hardware.

## 7. Limitations (please read)

- No GPU measurements. All H100 numbers are first-order bounds from specs: no overlap, no kernel inefficiency, no inter-GPU communication; 70B rows assume a single device it does not fit on.
- The one measurement (CPU) shows the roofline is about 2x optimistic at the crossover.
- AMD MI300X, CXL and NVMe figures were not collected (sources not available).
- The queueing results assume Poisson background and deterministic service; the BurstGPT test has arrival times only.
- The MoE results assume uniform independent routing.
- The simulator's catalog multipliers and speed ratios from the first version are unvalidated.
- The literature check (section 8) used search-result abstracts, not full papers.

## 8. Prior work, so you can place this

Search of 2026-11-26, from summaries; the papers were not read in full and the search is not exhaustive.

- Decode attention being memory-bound at any batch size because each sequence has its own KV cache is stated in the literature ([Helix Parallelism, arXiv 2507.07120](https://arxiv.org/html/2507.07120v1)). My critical-context values are a closed form of a known conclusion.
- KV quantization becoming the larger lever at long context: [QuantSpec, arXiv 2502.10424](https://arxiv.org/pdf/2502.10424), [KV4-RL, arXiv 2610.14448](https://arxiv.org/pdf/2610.14448), [KV Pareto, arXiv 2512.01953](https://arxiv.org/html/2512.01953v1).
- Speculative KV prefetching from CPU memory is established: [InfiniGen, OSDI 2024](https://arxiv.org/html/2406.19707v1), [KVPR](https://arxiv.org/html/2411.17089v2). [Where Should the KV Cache Live?](https://arxiv.org/pdf/2609.16215) models speculative fetches competing with demand traffic and reports uniformly negative results for prefetch; [SP-MoE](https://arxiv.org/pdf/2510.10302) reports prefetch failures from bandwidth contention. My congestion finding is consistent with these, not new relative to them.
- I found no paper with a queueing-based break-even accuracy, and no public Lean 4 formalisation of a roofline or inference cost model (nearest: the Rust crate `aprender-contracts`, which uses Kani). If that holds up, a verified threshold/roofline library is a small engineering gap, not a scientific contribution.

## 9. Next step and how to help

The largest gap is a GPU measurement: step time versus batch and context (including MoE with speculative verification), and KV transfer time over the link. If you have access to a GPU, or know of real serving traces with KV-reuse information, I would like to hear from you. Corrections to any claim above are welcome as issues.

## 10. Reproduce

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # pytest
.venv/Scripts/python -m pytest -q                            # 73 tests
.venv/Scripts/python experiments/h6_sensitivity.py           # H6 in the simulator
.venv/Scripts/python experiments/h6_calibrated.py            # H6 from specs
.venv/Scripts/python experiments/h6_moe.py                   # MoE extension
.venv/Scripts/python experiments/h13_kv_prefetch.py          # H13 from specs
.venv/Scripts/python experiments/h13_sim_vs_formula.py       # H13 vs simulation
cd formal && lake exe cache get && lake build                # Lean (several GB of disk)
```

`experiments/measure_roofline.py` needs `requirements-bench.txt` (numpy). On Linux or macOS use `.venv/bin/python`.
