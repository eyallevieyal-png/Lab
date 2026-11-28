# Prioritized implementation roadmap

Ordered by value per unit of effort, and by what unblocks what. "Free" means no paid service (a standing constraint of the owner).

## Next three highest-value engineering tasks
1. **Run the T4 repeat protocol** (`experiments/run_gpu_repeats.py --runs 5`, two sessions) and commit `aggregated.json`. It turns the only hardware evidence from an anecdote into a number with an interval. Free; tooling done and tested on the stored files.
2. **H17 diagnosis, second step.** First step DONE (`experiments/h17_gate_diagnosis.py`): the gate trails a static threshold even with the true utilisation, so the load estimate is not the cause. Next: sweep lead, period and the accuracy estimator window, and compare the analytic stall model against the engine's actual stall for the same parameters, to find which assumption of the M/D/1 threshold breaks. This decides whether the "potentially independent contribution" in H13 survives as an online gate. Free; about a day.
3. **Make results self-describing and reproducible end to end:** store per-seed raw rows (including prefetch counts) in the benchmark JSON, record the BurstGPT URL and SHA-256 next to the two trace results, and add a CI job that runs `pytest`, `verify_simulator_fixes` and `make_results_summary` and fails if a committed result differs from a regenerated one. Free.

## Backlog

| # | Task | Depends on | Notes |
|---|---|---|---|
| 4 | H16 decay sweep and change detection for the `phase_shift` regression | none | cheap; tune on seeds 100-119, evaluate on 0-19 |
| 5 | Extend the engine to several servers; unit-test that one tenant equals the single-tenant engine | none | prerequisite for H19 |
| 6 | Look for a public MoE routing trace; if none, keep H18 synthetic and say so | none | |
| 7 | Serving harness (small model on a T4; 8B on an H100): prefill/decode step time, TTFT, P99, throughput on a fixed seeded trace | stage A | needed for hardware claims about orchestration |
| 8 | CUDA `TierBackend` (pinned host buffers, own-vs-foreign link load from NVML) and a test that it matches measured copies | 7 | first real backend behind the interface |
| 9 | Energy measurement (NVML energy counters) in `measure_gpu.py` | stage A | only if the GPU exposes it |
| 10 | Concurrent-interference benchmark (two processes on one GPU and link) | 7 | feeds H19 |
| 11 | Rebuild the Lean library in CI and publish the theorem count automatically | none | done once by hand on 2026-11-28 (success); the documented count was 20, the file has 22 |
| 12 | H100-class run of items 1-3 and 7-10 | owner decides how to get access | not free |
| 13 | CXL measurement | a CXL-capable server | until then the CXL tier is a model |
| 14 | Move source into a package layout (`src/` to `lab/`) and retire the `python -m` path hacks | none | optional; not done in this audit because it would break every import without adding evidence |

## Definition of done for any new claim
It has an experiment ID, a script that regenerates it, a stored raw output, a seed list of at least 20, a 95% CI, a label (simulated / measured / computed from specs / proved / assumed), a place in `EXPERIMENT_REGISTRY.json`, and a statement of what would falsify it.
