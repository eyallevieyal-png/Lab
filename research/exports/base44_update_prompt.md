# Prompt to paste into the Base44 editor chat (site: memory-frontier-labs)

Update the site content as follows. Keep the existing design, navigation and style. Do not invent numbers: use only the facts below. All facts come from the public repository https://github.com/eyallevieyal-png/Lab (article: docs/POST.md).

## 1. Correction banner (top of Home, Dashboard and "72-Run Results" pages)

Add a prominent notice (the repository link in it may be added only after the audit branch is merged to main), visible without scrolling:

"CORRECTION (Nov 2026): An audit of the simulator behind the 72-run evaluation found 8 defects that made its headline results artifacts (for example, the 'Reactive' controller returned the Oracle's answer, and 'regret' was a fixed fraction of cost). The results on this page were produced before the audit and are SUPERSEDED. They are kept for transparency only. After the fixes, some conclusions flipped: prediction does not beat data-local placement unless it is gated. See the audit write-up on GitHub."

Link "audit write-up" to https://github.com/eyallevieyal-png/Lab/blob/main/docs/POST.md

On the "72-Run Results" page, label every number "superseded (pre-audit simulator)". Replace the home "CORE RESULT" tile (BitNet + speculative decoding, P99 -24.6%, throughput +13.0%) with a small "Superseded pre-audit result" tile, and move the new findings (section 3) to the main tile position.

## 2. Research status block (Home)

- EXPERIMENTAL STATUS: change "Planned — simulator in design" to "Simulator audited and fixed (128 tests); 22 Lean 4 theorems (library builds); first GPU measurement on one free-tier Tesla T4"
- INTEGRITY: "Every claim is labelled proved / simulated / computed from specs / measured / assumed"
- LATEST DISCOVERY (replace text), dated 2026-11-28:
  "The first version of our simulator was wrong, and fixing it changed the conclusions. In the corrected work: (1) single-stream decode is memory-bound by 50-300x on H100 specs, but prefill becomes compute-bound at about 300 tokens, so the 'compute-memory inversion' (H6) holds per phase, not globally; (2) for prefetching an offloaded KV cache there is a closed-form accuracy threshold p* = (u + gamma) / (1 + u), proved in Lean 4; (3) for mixture-of-experts, the batch size where decode becomes compute-bound is about (total / active parameters) times that of a dense model (9.9x for Qwen3-30B-A3B, 3.7x for Mixtral-8x7B, computed, assuming uniform routing); (4) first GPU measurement (one free Colab Tesla T4 run): the roofline predicts the compute/memory crossover within the resolution of the batch grid, is about 1.4x optimistic in the memory-bound regime, and decode attention time grows in proportion to KV bytes."
  Under it, in small text: "One GPU, one run: a T4 is not an H100. This is not a breakthrough; several conclusions are known in the literature. The contribution is the method: audit the baseline, separate proved from simulated from measured, publish the limits."

## 3. Update H6 card ("Compute–Memory Inversion")

Keep the claim about relative bottleneck composition and add: "Proved in an additive cost model (Lean 4). On public specs it holds for decode and fails for prefill (compute-bound from ~300 tokens). bf16 to fp8 on H100 moves the balance point by exactly 1.00x. Weight-only int4 makes decode less memory-bound, the opposite of H6's premise."

## 4. Add a section "How we label claims"

Table with five rows:
- Proved: Lean 4 + Mathlib theorem, compiles, no sorry, standard axioms only; true inside the stated model.
- Simulated: output of the simulator; depends on assumptions, not on hardware.
- Computed from specs: first-order roofline from vendor numbers and public model configs; an ideal bound, not a measurement.
- Measured: actually timed: one laptop CPU, and one run on a Colab Tesla T4 (not an H100).
- Assumed: a chosen parameter (data size, link utilisation, routing).

## 5. Add a section "What is still missing" (limitations)

- No H100-class measurement.
- No measurement of mixture-of-experts with speculative decoding.
- No measurement of a contended host-device link (the T4 environment moved KV at only ~1 GB/s, probably a Colab host limit).
- All GPU results are from a single run.
Next steps: an H100-class GPU measurement with a profiled attention kernel; a contended-link test of the H13 prefetch threshold; MoE with speculative verification; a repeat on another session or card.

## 6. Links

Add a visible "Code, tests and Lean proofs on GitHub" button on Home and About pointing to https://github.com/eyallevieyal-png/Lab
Update the "Research Radar" / "Research Journal" with one entry dated 2026-11-28: "Repository audit published: superseded results archived, simulator defects verified fixed in code, orchestrator prototype (simulated), first GPU measurement."

## 6b. Add a section "Orchestrator prototype (simulated)"

Text: "A working software prototype of an adaptive memory orchestrator (workload telemetry, a causal predictor, placement and prefetch policies over simulated HBM / DRAM / CXL / NVMe tiers) was benchmarked against an LRU baseline on four synthetic workloads, 20 seeds each, with 95% confidence intervals. All tier parameters are assumptions and CXL is a model only: this is not evidence about physical hardware. Findings: (1) prediction roughly halves P99 latency on a predictable workload and can double it on an unpredictable one when the link is busy; (2) a congestion-aware prefetch gate derived from the proved threshold avoids the bad case but is worse than a simple accuracy-above-0.5 rule on the predictable workload, and a diagnosis shows the cause is the threshold model, not the load estimate; (3) reuse-score placement helps on Zipf-distributed access (-10% to -15% P99) and hurts on a phase-shifting working set (+46% P99). An oracle that knows the next request would reduce P99 by 57% to 80%, so predictor quality is the main limit."
Link "orchestrator prototype" to https://github.com/eyallevieyal-png/Lab/blob/main/research/RESEARCH_STATUS.md (after the branch is merged to main; until then do not add the link).

## 7. Remove or soften

Remove the phrase "Planned — simulator in design". Do not claim novelty. Do not claim that results are absolute measurements.
