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
* H6 probe: operand reads take 92-96% of compute + memory time in every precision and workload, and the share falls slightly with lower precision.
  That is the opposite of an "inversion" and depends on the working-set scale above. It is not evidence against H6 in general.

## Not validated

Nothing here was compared with measurements. The old `results_summary.json` at the repository root and the old files in `experiments/results/`
come from the first version; the files in `experiments/results/` were regenerated, `results_summary.json` was not.
