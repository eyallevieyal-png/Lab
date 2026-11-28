# Research audit (2026-11-28)

Scope: the whole repository at commit `7c17e4e` (main). Method: read every script and result file, map which script writes which output, re-run what can be re-run, check documented numbers against stored data, and check the eight simulator defects with executable tests. What was **not** audited is listed in section 8.

## 1. Dependency map (script -> output)

| Script / module | Writes | Depends on | Re-run in audit |
|---|---|---|---|
| `run_smoke.py` -> `src.simulator.smoke_run` | `experiments/results/smoke_summary.json` | `src/*` | yes; only difference is the new field `prediction_cost_charged` |
| `run_smoke.py` -> `generate_research_report` | `experiments/results/research_report.json` | `src/*` | yes; byte-identical |
| `research/pipeline/make_results_summary.py` (new) | `results_summary.json` (root) | `src/*` | new |
| `experiments/h6_calibrated.py` | `h6_calibrated.json` | `calibration/hardware_models.py` (public specs) | yes; byte-identical |
| `experiments/h6_sensitivity.py` | `h6_sensitivity.json` | `src/*` | yes; byte-identical |
| `experiments/h6_moe.py` | `h6_moe.json` | `calibration/moe_model.py` | yes; byte-identical |
| `experiments/h13_kv_prefetch.py` | `h13_kv_prefetch.json` | `src/kv_prefetch.py` | yes; byte-identical |
| `experiments/h13_sim_vs_formula.py` | `h13_sim_vs_formula.json` | `src/kv_prefetch.py` | yes; byte-identical |
| `experiments/h13_analytic_vs_sim.py` | `h13_analytic_vs_sim.json` | `src/kv_prefetch.py` | yes; byte-identical |
| `experiments/h13_trace_background.py` | `h13_trace_background.json` | **external BurstGPT CSV** (not in repo, hash not recorded) | **no** |
| `experiments/h13_online_controller.py` | `h13_online_controller.json` | same CSV | **no** |
| `experiments/measure_roofline.py` | `measured_roofline.json` | the machine it ran on | no (hardware timing) |
| `experiments/measure_gpu.py` | `measured_gpu_*.json` | a CUDA GPU | no (no GPU here) |
| `experiments/analyze_gpu.py` | prints derived numbers | `measured_gpu_*.json` | covered by tests |
| `orchestrator/bench.py` (new) | `research/results/orchestrator_bench.json`, dashboard HTML | `orchestrator/*`, `src/kv_prefetch.py` | new |
| `experiments/h17_gate_diagnosis.py` (new) | `h17_gate_diagnosis.json` | `orchestrator/*` | new |
| `formal/LabFormal/Orchestration.lean` | (proofs) | Lean 4 + Mathlib | built: success |

## 2. Authoritative engine and superseded engines

- **Authoritative:** `src/simulator.py` (+ `memory.py`, `workloads.py`, `config.py`) since commit `b702005` (2026-11-25, "Fix simulator validity"), extended by `6ce6419` (sensitivity hooks) and by this audit (one added output field). `src/kv_prefetch.py` is the authoritative queueing model for H13. `orchestrator/` is the authoritative engine for the policy comparison (EXP-ORCH-001).
- **Superseded:** the first simulator, commit `feca1e6` (2026-09-17). Moved to `archive/legacy_simulators/v0_2026-09-17/` as `.py.txt`. No other simulator generations exist in the history.
- The repository never contained a simulator for the "72-run" scenario set (FP16/BitNet/INT4 with and without speculative decoding, 4 controllers). See finding F1.

## 3. The eight defects: verified in the current execution path

`python -m research.validation.verify_simulator_fixes` exercises the code that experiments import and exits 0 only if all eight checks pass. Result: **8 of 8 pass** (`research/validation/simulator_fixes_report.json`).

| # | Defect | Check | Evidence from the run |
|---|---|---|---|
| D1 | `hash()` seeding | workload stream compared under two `PYTHONHASHSEED` values in separate processes | identical |
| D2 | Reactive returned the oracle | source of `ReactiveController.select_resource` has no reference to the oracle; behaviour differs from the oracle | differs on 20 of 48 operations |
| D3 | Fake regret | oracle regret 0 for every operation; regret >= 0 for three other controllers | 24 and 72 operations |
| D4 | Memory cost unused | total memory time differs between fp16 and int4 | 163.2 vs 76.2 ms |
| D5 | Prediction cost never charged | `prediction_cost_charged` > 0 for Predictive, 0 for Reactive, included in `controller_time` | yes. **Finding F3:** the older field `prediction_cost` is still reported for every controller; it is now documented and a charged field was added |
| D6 | Calibration cannot vary | ECE with offset 0 vs -0.3 | 0.050 vs 0.318 |
| D7 | Seed ignored | seed 1 vs 2 streams differ | yes |
| D8 | Bandwidth / 8, trivial sizes | 1 GiB over HBM = 0.299 ms; typical DRAM->HBM transfer | 1.57 ms |

## 4. Findings

| ID | Finding | Severity | Action taken |
|---|---|---|---|
| F1 | The "72-run" results (BitNet speculative P99 -24.6%, throughput +13%, FP16 -17.7%, INT4 at 80% acceptance -24.4%) exist in this repository only as prose in `Master_promt.md` section 34 ("Historical experimental baseline"). No code and no raw data produce them. The public research website presents them as the core result | **High** (unsupported headline claim) | documented as not reproducible; registered as `LEGACY-72RUN-001` (superseded); website text prepared (`research/exports/`), not deployed |
| F2 | Root `results_summary.json` was the output of the defective v0 simulator and was never regenerated. Provenance check: oracle mean regret 0.1429 (an oracle has 0), Reactive p99 identical across seeds, 2 seeds, "7 passed" tests | **High** | archived as `SUPERSEDED__results_summary.json`; replaced by a regenerated file with metadata |
| F3 | Record field `prediction_cost` is reported for controllers that are not charged for it | Medium (misleading, not a calculation error) | added `prediction_cost_charged`; comment on the old field |
| F4 | Tests wrote into tracked result files (`test_report.json` is byte-identical to `research_report.json`; `test_smoke.json` is a copy of the smoke output) | Medium (duplicates, silent overwrites) | tests now use `tmp_path`; the duplicates were removed from the active tree (they are regenerated copies; history keeps them) |
| F5 | Documented Lean theorem count (20) disagrees with the file (22 `theorem` declarations; the article's own table lists 22 names) | Low | corrected in README, POST.md, REPORT.he.md with a visible correction note. The library was rebuilt (`lake build`, 18,295 jobs, no errors, no `sorry`); axioms were not re-listed |
| F6 | Test counts differ across documents (61, 73, 82, 84) | Low | updated to 128 with a date |
| F7 | `README.md` and `Readme.md` both tracked: a case collision on Windows made `README.md` show as modified and unstageable | Medium (blocked normal workflow) | `Readme.md` (an earlier master prompt, 1,481 lines) archived to `archive/legacy_reports/`; `README.md` rewritten |
| F8 | Two H13 results depend on the BurstGPT CSV, whose URL is in a docstring but whose version and SHA-256 are not recorded | Medium (not reproducible as stored) | registered with that status; roadmap item added |
| F9 | `Master_promt.md` (1,709 lines) mixes specification, historical numbers and instructions; it is easy to read its numbers as results | Low | left in place (it is a project document); README now points to the registry as the source of truth |
| F10 | No plots existed in the repository (no PNG/SVG/HTML), so nothing obsolete had to be removed from active presentation; the public site's charts are outside the repo | Info | three figures regenerated from stored data (`research/figures/`) |
| F11 | Orchestrator benchmark: the first version fed the H13 gate the TOTAL link utilisation, which double-counts the orchestrator's own traffic (the formula's `u` is background load). Fixed before the reported results; both versions are described in `RESEARCH_STATUS.md` | Info (own bug, found and fixed during this work) | `foreign_only` utilisation |

## 5. Core research tracks: what the evidence supports and where it stops

### H6: compute-memory inversion
- **Proved (Lean, additive model):** memory share `M/(C/k+M)` is strictly increasing in the compute speed-up k. True inside the model.
- **Computed from specs (H100 SXM page, Llama 3.1 8B/70B configs):** single-stream decode is 50-300x memory-bound; the KV cache is read per sequence, so often there is no balance batch at all. Prefill becomes compute-bound at a *configuration-dependent* number of tokens: S* = 316 (8B, bf16), 300 (70B, bf16), **80 (8B, int4 weights-only with dequantisation)** (`h6_calibrated.json`). The 80 shows that the "about 300 tokens" figure must not be generalised beyond the tested configurations. Prefill and decode are kept separate everywhere.
- **bf16 -> fp8 on H100** moves the balance point by exactly 1.00x (a consequence of the model: compute peak doubles while bytes halve).
- **Measured:** laptop CPU roofline (accurate at B=1 and B>=256, 3.3-3.7x slower in between) and one T4 run (about 1.4x optimistic in the memory-bound regime; crossover predicted 92, measured 64-128). One machine each.
- **Not established:** arithmetic-intensity effects of real kernels, quantisation kernels' real efficiency, multi-GPU, any H100 measurement.

### H13: predictive KV prefetching
- **Proved (Lean):** `p* = (u+gamma)/(1+u)` under its assumption that a wrong prefetch delays the demand fetch with probability u and a right one hides the whole transfer.
- **Simulated (queueing, FIFO link, Poisson background):** the formula matches up to u of about 0.5 and breaks above; the congestion-aware M/D/1 form is closer. **Mathematical derivation and experimental validation are different things here:** the theorem is correct, the model it describes stops matching the simulation under load.
- **New in this audit (simulated):** applied as an online gate inside the orchestrator, the congestion-aware threshold is *worse* than a static accuracy threshold of 0.5 on a predictable workload under background load, and equal on unpredictable ones. A diagnosis (seeds 100-119) shows the gap persists with the true utilisation and with load ignored (u=0), so the load estimate is not the main cause. Mechanism not isolated. See `RESEARCH_STATUS.md`.
- **Not established:** behaviour on a real link, real KV reuse traces, prediction accuracy of a real predictor on real sessions, contended links on hardware.

### MoE
- **Theoretical estimates, original assumptions:** the batch size where decode becomes compute-bound is about (total/active parameters) times the dense twin's: **9.9x (Qwen3-30B-A3B) and 3.7x (Mixtral-8x7B)** with 8-position speculative verification at context 512, uniform independent routing, bf16 (`h6_moe.json`: B* = 1,612/482 vs 163/49, and 331/148 vs 90/40). With plain decode (one position) there is no balance batch at all.
- **Not established:** expert activation patterns, expert reuse, routing imbalance, expert-weight movement, KV requirements and link contention for MoE. **No MoE measurement exists.** H18 starts from synthetic skew and says so.

## 6. Claims audit: public and documentation text
| Statement | Where | Verdict |
|---|---|---|
| "Core result: BitNet + speculative decoding, P99 -24.6%, throughput +13%" and the six-scenario table | public website (base44) | **unsupported by this repository** (F1). Must be marked superseded |
| "EXPERIMENTAL STATUS: Planned, simulator in design" | public website | **outdated**: a corrected simulator, proofs and one GPU measurement exist |
| "276 GB/s, 24.9 TFLOPS, about 1 GB/s KV link" | POST.md, task brief | matches `measured_gpu_tesla_t4.json` (276.1, 24.93; host-to-device 0.99-1.23 and device-to-host 1.45-2.12 GB/s). The 1 GB/s is the Colab host, not a T4 property |
| "20 Lean theorems" | README, POST.md, REPORT.he.md, website draft | **wrong count; corrected to 22** (F5) |
| "9.9x and 3.7x" | POST.md | correct arithmetic of the stored data; **theoretical**, uniform routing |
| "prefill compute-bound from about 300 tokens" | POST.md | true for two bf16 configurations; 80 for int4 weights-only; configuration-specific |
| Literature comparison (POST section 8) | POST.md | not re-checked in this audit (abstract-level in the original) |

## 7. Leakage and statistics review
- **Orchestrator predictor:** online, updated only with past requests; its accuracy is scored only against predictions made before the outcome; only the clairvoyant bound sees the future. Unit-tested.
- **Seeds and intervals:** benchmark 20 seeds, t-based 95% CI, paired differences on identical traces; diagnosis uses separate seeds (100-119). No multiple-comparison correction in the benchmark (stated on the dashboard); the registry and `NEXT_EXPERIMENTS.md` require Holm for new claims.
- **Older simulator results** use 5 seeds (controller comparison) and 2-3 seeds (H13 checks) without confidence intervals: adequate for the qualitative statements made, not for effect-size claims.
- **Orchestrator scenarios are generated by the author.** A policy that wins on them may not win on real traces; this is stated in every result.

## 8. Not audited / not done
- Axiom list of the Lean theorems was not re-printed in this session.
- The literature citations were not re-fetched.
- `h13_trace_background` and `h13_online_controller` were not re-run (external CSV).
- No GPU was available: the T4 repeat protocol is prepared, not run. No H100 or CXL work was possible.
- The public website was not modified (no access to its editor); replacement text is prepared.
