# Legacy simulator v0 (commit feca1e6, 2026-09-17)

**Status: SUPERSEDED. Not scientifically usable for conclusions about controllers, regret or prediction.**
Replaced by `src/simulator.py` (commit `b702005`, 2026-11-25, "Fix simulator validity"). The current path is checked by `python -m research.validation.verify_simulator_fixes`.

Files are stored as `*.py.txt` (not importable). `tests/` holds its 7 original tests, which passed while the simulator was wrong: they checked that code ran, not that results were valid.

## Which defects affected it

| # | Defect in v0 | Evidence in this folder |
|---|---|---|
| D1 | Workloads seeded with Python `hash()` (per-process random) | `workloads.py.txt` |
| D2 | `ReactiveController` returned `op["oracle_resource"]` | `simulator.py.txt`: 10 places return `op["oracle_resource"]` |
| D3 | Regret was a fixed fraction of the chosen resource's own time | oracle mean regret 0.1429 > Reactive 0.1409 in the archived report (impossible for a real regret) |
| D4 | `memory_cost` computed, never used | H6 could not be tested |
| D5 | Prediction cost never charged (`hasattr(controller, "predict")` always False) | `simulator.py.txt` |
| D6 | Prediction correct with probability exactly equal to its confidence | miscalibration impossible |
| D7 | `seed` did not change the workload | Reactive p99 identical for seeds 7 and 11 in the archived report |
| D8 | Bandwidth in GB/s divided by 8; sizes so small that transfers cost about 0 | `memory.py.txt` |

## Can any historical result still be used?
No result from this simulator supports a current conclusion. Reusable as design inputs only: the workload names, the precision catalog and the default tier parameters (they were kept in the current code and are labelled "assumed" there).
