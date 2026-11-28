# Prioritized implementation roadmap

Ordered by value per unit of effort, and by what unblocks what. "Free" means no paid service (a standing constraint of the owner).

## Next three highest-value engineering tasks
1. **Run the T4 repeat protocol** (`experiments/run_gpu_repeats.py --runs 5`, two sessions) and commit `aggregated.json`. It turns the only hardware evidence from an anecdote into a number with an interval. Free; tooling done and tested on the stored files.
2. **H17 follow-up.** Steps 1-2 DONE (the gate had three separable defects; the fixed gate wins in a narrow regime on held-out seeds). Next: (i) a steady-state latency metric so the overloaded regime (u = 0.75) stops depending on run length; (ii) why the fixed gate still trails a static threshold at accuracy 0.7, u = 0.5; (iii) the same comparison on a real arrival trace (BurstGPT gaps) and a real predictor. Free; about two days.
3. **Make results self-describing and reproducible end to end.** DONE: per-seed raw rows (including prefetch counts) are stored in the benchmark JSON; `research/check_all.py` regenerates the cheap results and fails if a committed one differs. STILL OPEN: record the BurstGPT URL, version and SHA-256 next to the two trace results (needs one download of the CSV), and decide whether to enable a CI job (a repository change the owner should approve). Free.

## Backlog

| # | Task | Depends on | Notes |
|---|---|---|---|
| 4 | H16 follow-up: causes of the two regressions left in `adaptive_v2` (unstructured traffic, shifting set under load); change detection | none | cheap; tune on seeds 100-119, evaluate on 0-19. The decay sweep and the demotion/eviction ablation are DONE |
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
