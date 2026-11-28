# Adaptive Memory Orchestrator: architecture

**Status: a working software prototype with SIMULATED tiers.** It demonstrates a control loop (telemetry, prediction, decision, transfer) and lets policies be compared reproducibly. It is **not** evidence of physical performance, and nothing here has run on CXL hardware. CXL appears only as a modelled tier with assumed numbers.

Code: `orchestrator/` · tests: `tests/test_orchestrator.py` · benchmark: `python -X utf8 -m orchestrator.bench` · results: `research/results/orchestrator_bench.json` · dashboard: `research/figures/orchestrator_dashboard.html`.

## 1. The loop

```
 request stream (sessions with KV blocks)
        |
        v
 Workload telemetry ---- observed: arrivals, accesses, link utilisation (foreign + own), queue wait
        |
        v
 Predictor (causal, online) ---- next-session guess + measured running accuracy p_hat
        |
        v
 Adaptive orchestrator (policy)
   |-- KV cache management : which blocks leave HBM (eviction victim)
   |-- Memory placement    : which lower tier an evicted block goes to (demotion target)
   |-- Predictive prefetch : fetch the predicted next session now?  (gated by the H13 threshold)
   '-- Bandwidth control   : skip a prefetch when the link queue is long
        |
        v
 TierBackend (interface) ---- reserve_read(src, bytes, now) -> completion time
        |                      utilization(src, now, foreign_only) ; queue_ms(src, now)
        v
 HBM  <-  DRAM  <-  CXL (model only)  <-  NVMe          (reads flow towards HBM)
```

## 2. Components and where they live

| Component (requested) | Module | Notes |
|---|---|---|
| Workload generator | `workload.py` | 4 scenarios: `agentic_loop` (predictable), `zipf_iid`, `uniform` (no structure), `phase_shift`. Seeded, deterministic. |
| Memory tier abstractions | `tiers.py` (`TierSpec`, `TierBackend`) | Parameters are labelled `assumed` / `spec` / `measured`. |
| Telemetry collection | `telemetry.py` (`Context`, `MarkovPredictor`) | Policies see a `Context` snapshot only; the future is visible only to a policy declared `clairvoyant` (tested). |
| Transfer-cost estimator | `estimator.py` | queue wait + latency + bytes / bandwidth; `stream_ms` is the `t_x` of the H13 threshold. |
| Placement controller | `engine.py` + `policies.py` | Engine owns capacity bookkeeping; the policy chooses victims and demotion tiers. |
| Adaptive decision engine | `policies.py` (`AdaptiveGate`, `Adaptive`) | Uses `src.kv_prefetch.analytic_breakeven` (the congestion-aware M/D/1 form of `p* = (u+gamma)/(1+u)`). |
| Baseline policy | `policies.py` (`LRUDemand`) | Plain tiered cache: LRU, waterfall demotion, demand fetch only. |
| Adaptive policy | `policies.py` (`Adaptive`) | Reuse-score eviction and demotion + gated prefetch. |
| Reproducible benchmark | `bench.py` | 20 seeds, 95% CI (t), paired differences on identical traces. |
| Comparison dashboard | `dashboard.py` | One self-contained HTML file, no scripts, no network. |

## 3. Decision inputs

| Input requested | In the prototype | Gap |
|---|---|---|
| Data importance | not implemented | needs semantic priority (system prompt, shared prefix, SLA class) from the serving layer |
| Reuse probability | first-order Markov next-session probability and measured top-1 accuracy | weak for non-sequential patterns; no per-block model |
| Access frequency | exponentially decayed access count per session | decay constant fixed (0.97 per request), not tuned or adapted |
| Time-to-next-use | observed mean inter-arrival time (coarse) | no per-session model |
| Data size | constant 32 MiB blocks, 8 per session | variable sizes and paged attention not modelled |
| Transfer cost | `TransferCostEstimator` | writes (demotion) are free and off the critical path: an assumption |
| Available bandwidth | tier spec; measured link utilisation (foreign = total minus own) | real backend must compute foreign load from counters |
| Queue depth | `queue_ms` per link | |
| Expected latency benefit | H13 gate: prefetch iff `p_hat > breakeven(u, gamma, lead, period)` and queue <= 2 transfers | `gamma` is assumed (0.02) |

## 4. Tiers (what is known, what is assumed)

| Tier | Prototype parameters | Evidence | What would change with real hardware |
|---|---|---|---|
| HBM | 64 blocks (scaled down), 3350 GB/s | vendor spec (H100 SXM page); capacity scaled | real capacity is far larger, but KV for long contexts is larger still |
| DRAM (host) | 192 blocks, 64 GB/s link into HBM, 1 us | **assumed** (PCIe Gen5 x16 class). The only related measurement is the Colab T4 host link: about 1.0-1.2 GB/s host-to-device and 1.4-2.1 GB/s device-to-host, which is 30-60x lower than the assumption | link bandwidth, pinned vs pageable memory, NUMA placement |
| CXL | 512 blocks, 32 GB/s, 2 us | **assumed, model only.** No CXL device was available or tested. Real behaviour depends on topology (direct-attached, switched, pooled), device and kernel tiering | everything: latency, bandwidth, contention with CPU traffic |
| NVMe | unbounded, 7 GB/s, 100 us | **assumed** | queue depth, read amplification, filesystem, GPUDirect Storage availability |

The architecture keeps the design separate from untested capability: the policy code never mentions a tier by capability; it sees `TierSpec` values and a backend. A CXL tier can be removed from `DEFAULT_TIERS` without changing a policy.

## 4b. Prototype simplifications (all bias the simulation, listed so they can be challenged)
Single serving thread; prefill output appears in HBM at first touch with no cost; demotion writes are free; a session's blocks are fetched as a unit per request; FIFO read channels per source tier; one background tenant (Poisson transfers of 4 ms) on one link; no compute/transfer overlap inside a request; no energy model.

## 5. Real hardware backends (to implement behind `TierBackend`)

1. **CUDA host-device backend**: pinned host buffers and `cudaMemcpyAsync` on a dedicated stream; foreign load estimated as measured link throughput (NVML PCIe counters) minus own bytes. Validate against the T4/H100 numbers from `experiments/measure_gpu.py`.
2. **NUMA/CXL backend**: allocate on a CXL-backed NUMA node (`libnuma`/`mbind`); measure bandwidth and latency per node with the same harness. Needs a CXL-capable server. Until then CXL stays `assumed`.
3. **NVMe backend**: `io_uring` or GPUDirect Storage where available.
The engine's contract with a backend is three methods; the first thing to test on any real backend is that `reserve_read` completion times match measured copies within the run-to-run spread (`experiments/aggregate_gpu_repeats.py` gives that spread).

## 6. What the benchmark currently shows
See `RESEARCH_STATUS.md` section 4: prediction helps a lot when the workload is predictable, the congestion-aware gate protects against catastrophic prefetching on unpredictable workloads (but a static accuracy threshold of 0.5 gives the same protection here and is better on the predictable workload), and reuse-score placement helps on Zipf traffic but hurts on a phase-shifting working set. All simulated.
