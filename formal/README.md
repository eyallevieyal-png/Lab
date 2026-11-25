# Formal core (Lean 4 + Mathlib)

`LabFormal/Orchestration.lean` proves, under explicit modelling assumptions, the mathematical core of H6 and H13.
It is mathematics, not evidence about hardware. Whether real workloads satisfy the assumptions is an empirical question
(see `docs/MODEL.md`).

| Theorem | Meaning |
|---|---|
| `netBenefit_pos_iff` | With `B + L > 0`, acting on a prediction pays iff `p > (L + c) / (B + L)` (H13 decision rule). |
| `netBenefit_at_threshold`, `netBenefit_mono_p`, `netBenefit_anti_c`, `reactive_optimal_below_threshold` | Net benefit is zero at the threshold, increases with reliability, decreases with cost; below the threshold staying reactive is optimal. |
| `memShare_strictMono`, `memShare_mem_Ioo` | Additive time model: the faster the compute, the larger the memory share (strictly), and it stays strictly between 0 and 1 (H6 in that model). |
| `memoryBound_iff` | Roofline: compute time <= memory time iff arithmetic intensity <= ridge point. |
| `regret_nonneg`, `regret_eq_zero_iff`, `oracle_lower_bound`, `totalRegret_ge_overhead` | Regret against the minimum-cost placement is never negative and is zero exactly for optimal placements. |

Checked: `lake build` succeeds (Lean v4.37.0-rc1, Mathlib v4.37.0-rc1), no `sorry`; every theorem depends only on
`propext`, `Classical.choice`, `Quot.sound` (`#print axioms` at the end of the file).

Not covered: that the simulator implements these statements (it is tested separately in `tests/test_validity.py`),
and anything about measured hardware.

Build: install elan, then `cd formal && lake exe cache get && lake build` (several GB of disk for Mathlib).
