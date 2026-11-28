"""Policies: the decisions the orchestrator makes. Same engine, same predictor, same traces: only these decisions differ.

  lru_demand         baseline: LRU everywhere, waterfall demotion, no prefetch (what a plain tiered cache does)
  always_prefetch    LRU + prefetch the predictor's top-1 every request
  static_threshold   LRU + prefetch when measured predictor accuracy > 0.5 (a fixed rule)
  adaptive_gate      LRU + prefetch only when accuracy exceeds the congestion-aware H13 break-even for the CURRENT link load and queue (H17 candidate)
  adaptive           adaptive_gate + reuse-score eviction and demotion placement (H16 + H17 candidate)
  oracle_prefetch    clairvoyant UPPER BOUND: prefetches the true next session. Not realisable; shown only to bound what any predictor could achieve.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from src.kv_prefetch import analytic_breakeven

from .estimator import TransferCostEstimator
from .telemetry import Context
from .tiers import LOWER

Block = Tuple[int, int]


@lru_cache(maxsize=4096)
def _breakeven(u: float, gamma: float, lead: float, period: float):
    return analytic_breakeven(u, gamma, lead, period)


class Policy:
    name = "base"
    clairvoyant = False

    def prefetch(self, ctx: Context, missing_by_tier: Dict[int, Dict[str, int]], est: TransferCostEstimator) -> Optional[int]:
        return None

    def hbm_victim(self, lru_order: List[Block], ctx: Context) -> Block:
        return lru_order[0]

    def lower_victim(self, tier: str, lru_order: List[Block], ctx: Context) -> Block:
        return lru_order[0]

    def demote_tier(self, block: Block, ctx: Context, free: Dict[str, int]) -> str:
        return next(t for t in LOWER if free[t] > 0 or t == "NVMe")


class LRUDemand(Policy):
    name = "lru_demand"


class _PrefetchBase(Policy):
    def _candidate(self, ctx: Context):
        return ctx.pred if ctx.pred is not None and ctx.pred != ctx.session else None


class AlwaysPrefetch(_PrefetchBase):
    name = "always_prefetch"

    def prefetch(self, ctx, missing_by_tier, est):
        return self._candidate(ctx)


class StaticThreshold(_PrefetchBase):
    name = "static_threshold"

    def prefetch(self, ctx, missing_by_tier, est):
        c = self._candidate(ctx)
        return c if c is not None and ctx.pred_accuracy > 0.5 else None


class AdaptiveGate(_PrefetchBase):
    """Prefetch only if p_hat > congestion-aware break-even(u_hat, gamma, lead, period) and the link queue is short (H13 threshold, applied online)."""
    name = "adaptive_gate"
    gamma = 0.02          # assumed predictor/controller cost as a fraction of one transfer
    margin = 0.02
    max_queue_tx = 2.0

    def prefetch(self, ctx, missing_by_tier, est):
        c = self._candidate(ctx)
        if c is None:
            return None
        need = missing_by_tier.get(c, {})
        if not need:
            return None                      # already resident: nothing to fetch
        tier = max(need, key=lambda t: est.stream_ms(t, need[t]))   # the slowest source decides the transfer time
        t_x = est.stream_ms(tier, need[tier])
        if ctx.queue_ms[tier] > self.max_queue_tx * t_x:
            return None                      # queue-aware: a long queue means a wasted prefetch would hurt the next demand fetch
        u = round(min(0.95, ctx.util[tier]) * 20) / 20
        lead = period = round(max(ctx.mean_gap_ms, 1e-6) / t_x * 4) / 4
        thr = _breakeven(u, self.gamma, lead, period)
        if thr is None:
            return None
        return c if ctx.pred_accuracy > thr + self.margin else None


class Adaptive(AdaptiveGate):
    """Adds placement: evict and demote by reuse score x refetch cost instead of recency alone."""
    name = "adaptive"

    @staticmethod
    def _score(ctx: Context, blk: Block) -> float:
        s = ctx.access_score.get(blk[0], 0.0)
        if ctx.pred == blk[0]:
            s += ctx.pred_accuracy        # predicted to be needed next: keep
        return s

    def hbm_victim(self, lru_order, ctx):
        return min(enumerate(lru_order), key=lambda ib: (self._score(ctx, ib[1]), ib[0]))[1]

    def lower_victim(self, tier, lru_order, ctx):
        return min(enumerate(lru_order), key=lambda ib: (self._score(ctx, ib[1]), ib[0]))[1]

    def demote_tier(self, block, ctx, free):
        ranked = sorted(ctx.access_score.values(), reverse=True)
        hot = self._score(ctx, block) >= (ranked[len(ranked) // 4] if ranked else 0.0)
        order = LOWER if hot else ("CXL", "DRAM", "NVMe")      # cold data goes to the slower tier first, keeping DRAM for the hot set
        return next(t for t in order if free[t] > 0 or t == "NVMe")


class OraclePrefetch(Policy):
    name = "oracle_prefetch"
    clairvoyant = True

    def prefetch(self, ctx, missing_by_tier, est):
        h = ctx.future_hint
        return h if h is not None and h != ctx.session else None


ALL_POLICIES = {p.name: p for p in (LRUDemand, AlwaysPrefetch, StaticThreshold, AdaptiveGate, Adaptive, OraclePrefetch)}
