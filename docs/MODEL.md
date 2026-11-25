# Model assumptions and changes from the first version

This simulator is a model, not hardware evidence. Every number it produces follows from the assumptions below. Branch `fix/simulator-validity`.

## What was wrong in the first version (found by reading the code and re-running it)

| # | Problem | Effect |
|---|---|---|
| 1 | `hash(name) % 17` seeded the workloads | Python randomises `hash()` of strings per process, so two runs of the same code gave different results (72 of 299 fields differed). |
| 2 | The Reactive controller returned `op["oracle_resource"]` | Reactive was the oracle with a small overhead, so "is prediction worth its cost?" was answered "never" by construction. |
| 3 | `oracle_time` was a fixed fraction of the chosen resource's own times | "Regret" was not regret: even the Oracle had positive regret (about 0.15) and in one case more than Reactive. |
| 4 | `memory_cost` was computed and never used | H6 (compute-memory inversion) could not be tested: no result could depend on the memory side of the precision. |
| 5 | `hasattr(controller, "predict")` was always False | The prediction cost was never charged to any controller. |
| 6 | Prediction was correct with probability exactly equal to the confidence | Calibration error could not occur, so it could not be studied. |
| 7 | `seed` did not change the workload | Reactive, Oracle and WSA gave identical results for every seed. |
| 8 | Bandwidths in GB/s were divided by 8 | Transfers were 8 times too slow. Sizes were also so small that transfer time was about zero. |

## The model now (all times in ms)

* Each operation has a `home_resource` (where its data lives). Running elsewhere first moves `size_bytes * memory_multiplier` bytes.
* `placement_cost(op, r)` = queue + transfer + preparation + execution + measurement + post-processing + synchronisation, where
  execution = compute + operand read time on `r` (or `max` of the two with `overlap_compute_memory=True`).
* Oracle = minimum placement cost. Regret = chosen placement cost - minimum (never negative, 0 for the Oracle).
  `total_regret` also includes the controller overhead and prediction cost.
* Reactive = data-local (never moves data, no access to the oracle).
* Predictive trusts the predictor. Adaptive acts on it only when `p*B - (1-p)*L - c > 0` (see `expected_net_benefit`).
  Confidence-gated adds a confidence threshold. WSA keeps its original rules, falling back to data-local.
* The predictor is right with probability `confidence + calibration_offset` (0 = calibrated by construction). `calibration_error` is measured (ECE).
* Useful / harmful / neutral are relative to staying data-local (placement cost lower / higher / equal).

## Assumptions that are choices, not facts

* Operand working sets are 64 MiB * (1 + (i % 9) * 0.35) * (1 + memory_pressure), with +-15% jitter. Results depend on this scale.
* The precision catalog multipliers, resource speed ratios, queue costs and QPU timings are unchanged from the first version and are unvalidated.
* `compute_ms_per_unit = 0.18` keeps the first version's compute scale, now read as milliseconds.
* The workload "state" feedback loop (+0.05 / -0.04) is unchanged.

## What the corrected model shows (simulation only, 5 seeds, 24 operations, int4)

* The Oracle is the lower bound in every run; Adaptive and Confidence-Gated Adaptive are within one standard deviation of Reactive in mean end-to-end time
  (they rarely act, and are never harmful); Predictive is clearly worse (12-18% harmful placements); WSA is worse than Reactive.
* So under these assumptions, prediction does not pay off against data-local placement unless it is gated, and gating makes it match Reactive, not beat it.
* H6 is analysed in the next section; the first probe (memory 92-96% of compute + memory time at the default scale) is one point of that sweep.

## H6 sensitivity (`experiments/h6_sensitivity.py`, results in `experiments/results/h6_sensitivity.json`)

For the data-local Reactive controller, memory share = M / (C/k + M) when compute is k times faster. The simulator matches this closed form exactly
(max error 0) and is strictly increasing in k in every configuration, as proved in `formal/LabFormal/Orchestration.lean` (`memShare_strictMono`).
The share reaches 50% at **k\* = C / M**, the compute speed-up needed for the "inversion". Measured k\* (fp16, int4; all workloads, 3 seeds):

| working-set scale | HBM | DRAM | CXL | NVMe | QPU |
|---|---|---|---|---|---|
| 1/256 | 2.3 / 2.1 | 0.53 / 0.48 | 0.36 / 0.32 | 0.23 / 0.21 | 1.1 / 1.0 |
| 1/16 | 2.3 / 2.1 | 0.51 / 0.47 | 0.34 / 0.31 | 0.20 / 0.20 | 1.1 / 1.0 |
| 1 (default, 64 MiB-scale) | 1.3 / 1.6 | 0.12 / 0.20 | 0.07 / 0.12 | 0.02 / 0.04 | 0.33 / 0.51 |
| 4 | 0.56 / 0.92 | 0.04 / 0.07 | 0.02 / 0.04 | 0.005 / 0.01 | 0.11 / 0.21 |

(k\* < 1 means memory already dominates at k = 1.)

* Only HBM (and the QPU at small sizes) is compute-dominated at baseline; DRAM, CXL and NVMe are memory-dominated at every size tested.
  For HBM at the default scale, a 1.3-1.6x compute speed-up already flips it.
* The precision catalog works against H6's premise: int4 shrinks memory bytes to 0.4 of fp16 but net compute only to about 0.89 (1.6 / 1.8),
  so quantization raises k\* (HBM 1.3 -> 1.57) instead of lowering it. These multipliers are unvalidated first-version values
  (4-bit weights would be 0.25 of the bytes), so this is a statement about the catalog, not about hardware.
* In the additive model H6 is a mathematical fact (proved). The empirical content is where k\* lies, i.e. the ridge point, which is set
  entirely by the working-set size, tier bandwidths and compute speeds. Those are assumptions here, so the simulator cannot confirm or refute H6 for real
  inference; it can only show which parameter regimes would.

## Not validated

Nothing here was compared with measurements. The old `results_summary.json` at the repository root and the old files in `experiments/results/`
come from the first version; the files in `experiments/results/` were regenerated, `results_summary.json` was not.

## H6 calibrated on public specifications (`calibration/`, `experiments/h6_calibrated.py`, `experiments/results/h6_calibrated.json`)

Inputs read on 2026-11-25: NVIDIA's H100 page (H100 SXM: 3.35 TB/s, 80 GB; H100 NVL: 3.9 TB/s, 94 GB; tensor throughput rows are footnoted "with sparsity",
so **dense = half of the printed figure is our derivation**) and the `config.json` files of Llama 3.1 8B and 70B (the computed parameter counts, 8.03B and 70.55B,
follow from those architectures). AMD MI300X could not be read (page timed out twice) and is not used. Nothing here is a measurement: it is a first-order roofline
calculation (no overlap effects, no kernel inefficiency, no communication, one device; 70B in bf16 does not fit one H100, so its rows are per-device idealisations).

One decode step, batch B, context T: compute time = B * (2 * matmul params + attention FLOPs) / dense peak; memory time = (weights + B * T * KV bytes per token) / bandwidth.

* **Batch 1 decode is memory-bound by a factor of about 50 to 300**: compute time is 0.3% to 2% of memory time for both models, every context length (512 to 32768),
  every scenario and both devices.
* **There is often no balance point at all.** The KV cache is read per sequence, so attention keeps the step memory-bound at any batch size whenever the
  attention part is heavier than the compute it saves (B* = "never"). On H100 SXM, Llama 3.1 8B has no balance point at any tested context.
  The only reachable balance points on H100 SXM are Llama 3.1 70B with int4 weights at context 512 (B* about 115, fits up to about 266 sequences);
  70B in fp8 at context 512 would need B* about 459 but only about 112 sequences fit.
* **H6's premise does not hold for fp8 on H100.** bf16 -> fp8 doubles the compute peak and halves the bytes, so the balance point moves by a factor of exactly 1.00.
  Weight-only int4 with bf16 compute does the opposite of H6: it shrinks bytes 4x with no compute gain, making decode less memory-bound (ratio x4, still far below 1).
  H6's "inversion" needs hardware compute to grow faster than bytes shrink; the data read here show that is not the case for this precision step.
* **The simulator's default scale is not decode.** Its HBM k\* (compute/memory time ratio at baseline) is 1.3 to 1.6, i.e. roughly balanced, while real decode on these devices is
  0.003 to 0.02. The default workloads therefore model a compute-heavy regime (large-batch or prefill-like), not single-stream decode.

Open: measured throughput (these are bounds from specifications), AMD and other accelerators, and prefill, where intensity is high and the conclusion may reverse.

### Prefill (same calibration, H100 SXM, one sequence, whole prompt at once)

Prefill becomes compute-bound at about 300 prompt tokens in bf16 and in fp8 (S* = 316 for Llama 3.1 8B, 300 for 70B); at 2048 tokens compute time is about 6.6-6.9
times memory time. Weight-only int4 lowers S* to about 75-80 tokens (even more compute-bound). So the picture is split: **decode is memory-dominated, prefill is
compute-dominated**, and H6's direction ("memory share grows as compute gets cheaper") has to be stated per phase. These are specification-based bounds, not measurements.

## A real measurement on one machine (`experiments/measure_roofline.py`, `requirements-bench.txt`)

Run on a laptop CPU (Intel Core i5-1035G1, 4 cores, 7.7 GB RAM, no GPU, numpy 2.5.5 / OpenBLAS, float32). Measured: about 13.7 GB/s memory bandwidth, about 222 GFLOPS float32 matmul,
ridge about 16 FLOP/byte. The test family `Y = W @ X` (W 8192 x 8192, 256 MiB) from matrix-vector (B = 1) to matrix-matrix (B = 512) against the roofline prediction
`max(flops/peak, bytes/bandwidth)`:

* B = 1: measured/predicted 0.94. B >= 256: 0.97-1.08. The model is accurate at both ends.
* B = 2 to 32: measured is **3.3-3.7 times slower than predicted** (time stays flat at about 65-74 ms while the model says about 20 ms), so in this regime the model is not predictive.
  A plausible but **unverified** cause is that the BLAS matrix-matrix kernel repacks the large matrix W before multiplying, moving about 3 times the counted bytes.
* The predicted balance batch is 33; the first measured batch reaching 50% of peak is 64. So the first-order roofline is optimistic by about 2x about where the crossover is.

What this supports: the memory-bound / compute-bound classification at the extremes. What it does not support: a quantitative prediction in the middle, or anything about GPUs.
Specification-based figures above should therefore be read as ideal bounds, with real kernels in the transition region likely slower than the model says.
