# Hardware validation plan (H20)

**Status: tooling prepared, no new hardware result produced.** This audit environment has no GPU (a laptop Intel CPU only), so nothing below has been run on a GPU in this session. No H100 or CXL number exists anywhere in this repository.

## 0. What exists today (preliminary, one machine, one session)

| Quantity | Value | Source | Caveat |
|---|---|---|---|
| Memory bandwidth, Tesla T4 | 276.1 GB/s (best of copy 240.8 and reduce 276.1) | `measured_gpu_tesla_t4.json` | one run, Colab, clocks varied (825-1395 MHz) |
| fp16 matmul peak, T4 | 24.93 TFLOPS | same | depends on matrix shape (square 8192); tall products underestimate |
| KV transfer in the Colab environment | host-to-device 0.99-1.23 GB/s, device-to-host 1.45-2.12 GB/s | same | virtualised host; not a hardware property of a T4 |
| Roofline ratio, memory-bound regime | about 1.4x optimistic (batch 2-64) | same | single run |
| Smoke run | `measured_gpu_tesla_t4_quick.json` (iters 5) | | cannot be pooled with the full run (the aggregator refuses it) |

Treat these as **preliminary measurements, not specifications.**

## 1. Stage A: reproduce the T4 numbers (free, user runs it on Colab/Kaggle)

```
# in a Colab T4 notebook, after cloning the repo
python experiments/run_gpu_repeats.py --runs 5 --cooldown 20
```
It runs `measure_gpu.py` five times in separate processes, then `experiments/aggregate_gpu_repeats.py` writes `experiments/results/repeats/<gpu>/aggregated.json`.
- **Do it in at least two separate sessions** (different days) to expose host-to-host variation.
- **Acceptance:** bandwidth and peak coefficient of variation <= 0.05 within a session; otherwise report the interval and the clock/power traces instead of a single number.
- **Pooling rules (enforced by code):** same GPU, torch and CUDA; quick runs excluded; at least 2 runs.
- **Methodology:** warmup then timed iterations with CUDA events; median reported next to mean; clocks and power recorded before and after each section (already stored by `measure_gpu.py`).

## 2. Stage B: a second, different GPU (free where possible)
Kaggle offers T4 x2 and P100 free; Colab free tier offers T4 only. A different card shows how much of the T4 behaviour (for example the 1.4x roofline optimism) is card-specific. Same script, same repeat protocol.

## 3. Stage C: H100-class validation (needs access that is NOT free)
Free tiers do not provide an H100. Options the owner must choose between (the standing rule is no paid services without explicit approval): a paid hourly rental, an institutional/academic allocation, or an employer/partner machine. **Do not start Stage C without that decision.** If access arrives, the package below is ready except the serving benchmark (section 4, items 4-7), which still has to be written and tested on a GPU.

### Measurements required
| # | Measurement | Tool | Status |
|---|---|---|---|
| 1 | GPU memory bandwidth | `measure_gpu.py` (bandwidth section) | **ready** |
| 2 | Actual host-device and (if present) peer transfers, pinned vs pageable | `measure_gpu.py` (kv_transfer section) extended with pageable/pinned and concurrent-copy cases | partly ready (idle link only) |
| 3 | Roofline ratio, decode attention, GEMM family | `measure_gpu.py` | **ready** |
| 4 | Prefill and decode step time on a real model, TTFT, P99, throughput | a small serving harness (HF/vLLM) with a fixed request trace | **not written** |
| 5 | KV cache movement under offload (move blocks to host and back during decode) | harness + `TierBackend` CUDA implementation | **not written** |
| 6 | Energy where measurable | NVML `nvmlDeviceGetTotalEnergyConsumption` around each section | **not written** (T4 power limit 70 W is recorded, not energy) |
| 7 | Concurrent workload interference | two processes sharing the GPU/link, one measured, one generating load | **not written** |

### Controls and statistics
- At least 5 full runs per configuration, in at least 2 sessions; report mean, 95% CI and CV; do not quote a number from fewer than 3 runs.
- Lock or record clocks (`nvidia-smi -q -d CLOCK,POWER`); discard or flag throttled runs rather than average them silently.
- Latency is reported as distributions (P50/P99), not means alone; the request trace is fixed by seed and stored.
- Prediction evaluation must be causal: the predictor never sees the evaluation requests (train/eval split by time; the prototype's predictor is online and tested for causality).
- Simulation and measurement are never pooled. A figure that shows both labels them.

## 4. Pre-registered reproduction criteria (what "the simulator's benefits reproduce" means)
Stated before any H100 data exists, so the outcome cannot be fitted afterwards.
1. **Roofline crossover:** measured balance batch within a factor of 2 of the predicted one (already true on the T4: predicted 92, measured 64-128; to be re-tested on the H100 with the H100's own measured bandwidth and peak).
2. **Memory-bound regime accuracy:** ratio of measured to predicted time inside the 1.0-1.5 band. A ratio outside it means the model needs an efficiency term, not that the architecture failed.
3. **H13 threshold:** on a real link with controlled background traffic, break-even accuracy measured at u = 0.25 and 0.5 within 0.1 of `analytic_breakeven`. A larger gap is reported as a model failure, as it was for the simple closed form above u = 0.5.
4. **Orchestrator benefit (H16/H17):** a significant P99 reduction (paired 95% CI excluding 0) of the adaptive policy over the LRU baseline on a trace with measured reuse structure. No claim of benefit if the CI includes 0.

## 5. CXL
No CXL hardware is available. The CXL tier stays a model until a CXL-capable server is accessible. When it is, measure per-node bandwidth and latency with the same harness before any orchestrator claim; the topology (direct, switched, pooled) is part of the result.

## 6. Reproducibility package contents
`experiments/measure_gpu.py`, `experiments/run_gpu_repeats.py`, `experiments/aggregate_gpu_repeats.py` (tested on the stored T4 files), `docs/GPU_MEASUREMENT.md` (user guide), this plan. Every result file stores GPU name, torch/CUDA versions, iterations and clocks.
