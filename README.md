> **Read this first.** The first version of this simulator had 8 defects that made its results artifacts; they are fixed on this branch and the old results should be discarded. The research write-up (decode vs prefill, KV-prefetch threshold, MoE, 20 Lean theorems, and the limits) is in [docs/POST.md](docs/POST.md); assumptions are in [docs/MODEL.md](docs/MODEL.md). No GPU measurements yet.

# Memory Frontier Labs Simulator

This repository reconstructs a minimal but scientifically structured simulator for the research described in [Master_promt.md](Master_promt.md).

## Scope

The simulator models:

- explicit memory tiers: HBM, DRAM, CXL, NVMe, and QPU
- compute and transfer costs
- workload diversity across agentic, branching, conversational, long-running, static-LLM, and tool-heavy patterns
- precision settings including FP16, INT4, BitNet b1.58, and BitNet a4.8
- reactive, predictive, workload-state-aware, adaptive, oracle, and confidence-gated adaptive controllers
- prediction confidence, cost, and placement decisions
- end-to-end latency and regret metrics

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 run_smoke.py
python3 -m pytest -q
```

## Structure

- src/config.py — configuration and precision catalog
- src/memory.py — explicit memory hierarchy and transfer model
- src/workloads.py — workload generators
- src/simulator.py — simulation engine and smoke experiment
- src/metrics.py — helper metrics
- tests/ — unit and integration checks

## Notes

This implementation is intentionally explicit and auditable rather than tuned to reproduce historical numbers. The historical results in the prompt are treated as reference points only, not as hard-coded outputs.
