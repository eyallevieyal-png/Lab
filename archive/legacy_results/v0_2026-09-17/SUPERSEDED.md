# SUPERSEDED — NOT VALID FOR CURRENT RESEARCH CONCLUSIONS

Produced by the defective simulator in `../../legacy_simulators/v0_2026-09-17/` (commit `feca1e6`, 2026-09-17).
`SUPERSEDED__results_summary.json` was the repository-root `results_summary.json` until the 2026-11 cleanup; it was never regenerated after the simulator was fixed.

Provenance checks (`research/validation/provenance_check.py`) against the current simulator:

| Quantity (mean over controllers rows) | archived v0 | current simulator |
|---|---|---|
| Oracle mean regret | 0.1429 (impossible: an oracle has zero regret) | 0.0000 |
| Reactive mean regret | 0.1409 (lower than the Oracle's) | 0.1143 |
| Seeds in the compared archive | 2 (7, 11) | compared on the same 2 seeds (the current pipeline uses 5) |
| Reactive p99 across seeds | identical for seeds 7 and 11 (seed had no effect) | differs (D7 check) |
| Verification recorded | "pytest: 7 passed" (tests checked execution, not validity) | 84 tests + 8 executable defect checks |

## The "72-run" figures quoted elsewhere

The headline numbers of the earlier public write-up (72 runs = 6 scenarios x 4 controllers x 3 seeds; BitNet+speculative P99 52.820 -> 39.800 ms, -24.6%; throughput +13%; FP16 speculative -17.7%; INT4 at 80% acceptance -24.4%) appear in this repository ONLY as prose inside `Master_promt.md` section 34 ("Historical experimental baseline"). There is no code and no raw data in the repository that produces them, and the current simulator has a different scenario set (6 workloads x 6 controllers). They are therefore **not reproducible from this repository** and must not be presented as current findings.
