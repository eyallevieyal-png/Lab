> **Read this first.** The first version of this simulator had 8 defects that made its results artifacts; they are fixed and the old results are archived as **superseded**. The "72-run" figures quoted in earlier public write-ups are not reproducible from this repository and are not current findings. The research write-up is in [docs/POST.md](docs/POST.md); the audit is in [research/RESEARCH_AUDIT.md](research/RESEARCH_AUDIT.md); what is currently trustworthy is in [research/RESEARCH_STATUS.md](research/RESEARCH_STATUS.md). One GPU measurement exists (a free-tier Colab Tesla T4); there is no H100, MoE, CXL or contended-link measurement yet.

# Memory Frontier Labs

Research thesis: *Adaptive AI Infrastructure: predictive orchestration of compute, memory and data movement in low-bit LLM inference.*
Integrity rule: every claim carries one label (below). Research integrity is more important than impressive numbers.

## Evidence classes (used in `research/EXPERIMENT_REGISTRY.json`)

| Class | Meaning |
|---|---|
| `reproduced` | re-run in the 2026-11-28 audit; the committed output was byte-identical |
| `validated_simulation` | simulator output; the simulator passes 8 executable defect checks and the test suite. **Simulated, not measured** |
| `preliminary_hardware` | real timings, one machine / one run (for example a Colab T4). Not a hardware specification |
| `theoretical` | calculation from public specs, or a proof (Lean 4) that holds inside its stated model only |
| `superseded` | produced by obsolete code; archived; **not valid for current conclusions** |
| `unverified_hypothesis` | designed, not run |

## Where things are

| Path | What |
|---|---|
| `src/` | the corrected analytical simulator (controllers, regret, prediction) and the KV-prefetch queueing model |
| `orchestrator/` | adaptive memory orchestrator prototype with simulated tiers (workload generator, tiers + backend interface, telemetry, policies, benchmark, dashboard) |
| `experiments/` | H6 / H13 / MoE experiment scripts, GPU measurement scripts, `results/` (the authoritative result files) |
| `calibration/` | spec-based hardware and MoE models |
| `formal/` | Lean 4 library (22 theorems) |
| `research/` | audit, status, registry, plans, validation and pipeline scripts, figures, exports |
| `archive/` | superseded simulator, results and reports, kept byte-for-byte for the record |
| `results_summary.json` | regenerated from the current simulator, with metadata (experiment id, simulator commit, config, date) |
| `docs/` | write-ups: `POST.md` (English), `REPORT.he.md` (Hebrew), `MODEL.md` (assumptions), `GPU_MEASUREMENT.md` (how to run the GPU script) |

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt            # pytest, numpy
.venv/Scripts/python -m pytest -q                                   # 128 tests
.venv/Scripts/python -X utf8 -m research.validation.verify_simulator_fixes   # the 8 defects, checked in the current execution path
.venv/Scripts/python -X utf8 -m research.pipeline.make_results_summary       # regenerates results_summary.json
.venv/Scripts/python -X utf8 -m orchestrator.bench                  # policy comparison, 20 seeds, about 3-5 minutes
.venv/Scripts/python -X utf8 experiments/h17_gate_diagnosis.py      # why the H13 gate loses to a static threshold (simulated)
.venv/Scripts/python -X utf8 -m research.figures.make_figures       # figures from stored data
.venv/Scripts/python -X utf8 -m research.registry.build_registry    # experiment registry
```
GPU measurement (needs a CUDA GPU, for example a free Colab T4): see [docs/GPU_MEASUREMENT.md](docs/GPU_MEASUREMENT.md) and `experiments/run_gpu_repeats.py`.
Windows note: this repository used to contain both `README.md` and `Readme.md`, which collide on case-insensitive file systems; the second one is archived.

## Current state in one paragraph

Prediction helps a lot when the workload is predictable and can hurt badly when it is not; a trivial accuracy threshold protects against the bad case as well as the closed-form H13 gate does, and the gate is worse on predictable traffic (simulated; orchestrator benchmark and its diagnosis). Memory-bound decode versus compute-bound prefill is a per-phase statement, true for the tested configurations only. The closed-form prefetch threshold is a proved theorem that stops matching a queueing simulation above link utilisation of about 0.5. A single T4 run shows the roofline about 1.4x optimistic in the memory-bound regime. Everything about H100, CXL and MoE performance is either computed from specs or unmeasured.

Code license: see `LICENSE`. Corrections and issues welcome.
