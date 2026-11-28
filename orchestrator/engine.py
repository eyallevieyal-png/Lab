"""Discrete-event engine for the adaptive memory orchestrator.

One request at a time is served: it needs every KV block of its session in HBM. Blocks missing from HBM are fetched over the source tier's link (FIFO channel,
shared with prefetches and with optional background traffic); the request stalls until the last block arrives, then computes. While a request runs, the
policy may prefetch the predicted next session. Latency = finish time - arrival time (so it includes queueing behind earlier requests).

Same arrivals, same trace and same background load for every policy of one seed (common random numbers), so differences between policies are not sampling noise.
All tier parameters are ASSUMED (see tiers.py). This is a software prototype of the control loop, not evidence about physical CXL or any real GPU.
"""
from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .estimator import TransferCostEstimator
from .policies import ALL_POLICIES, Policy
from .telemetry import Context, MarkovPredictor
from .tiers import DEFAULT_TIERS, LOWER, SimulatedBackend, TierBackend
from .workload import make_trace

Block = Tuple[int, int]


@dataclass
class Config:
    scenario: str = "agentic_loop"
    seed: int = 0
    n_sessions: int = 48
    session_blocks: int = 8
    block_bytes: int = 32 * 1024 * 1024
    compute_ms: float = 6.0
    period_ms: float = 14.0
    requests: int = 1500
    warmup: int = 100
    decay: float = 0.97
    bg_util: Dict[str, float] = field(default_factory=dict)   # background load per source tier, e.g. {"DRAM": 0.5}


def _pct(xs: List[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


class Engine:
    def __init__(self, cfg: Config, policy: Policy, backend: TierBackend | None = None):
        self.cfg, self.policy = cfg, policy
        self.backend = backend or SimulatedBackend(DEFAULT_TIERS, cfg.bg_util, seed=cfg.seed)
        self.est = TransferCostEstimator(self.backend)
        self.cap = {t.name: t.capacity_blocks for t in DEFAULT_TIERS}
        self.lru: Dict[str, "OrderedDict[Block, None]"] = {t.name: OrderedDict() for t in DEFAULT_TIERS}
        self.where: Dict[Block, str] = {}
        self.ready: Dict[Block, float] = {}
        self.pf: Set[Block] = set()
        self.pred = MarkovPredictor()
        self.score: Dict[int, float] = {}
        self.m = dict(hit=0, miss=0, moved=0, pf_bytes=0, wasted=0, busy={t: 0.0 for t in LOWER}, pf_issued=0)

    # ── tier bookkeeping ─────────────────────────────────────
    def _free(self) -> Dict[str, int]:
        return {t: (10**9 if self.cap[t] is None else self.cap[t] - len(self.lru[t])) for t in LOWER}

    def _insert_lower(self, blk: Block, tier: str, ctx: Context) -> None:
        cap = self.cap[tier]
        if cap is not None and len(self.lru[tier]) >= cap:
            victim = self.policy.lower_victim(tier, list(self.lru[tier]), ctx)
            del self.lru[tier][victim]
            self._insert_lower(victim, LOWER[LOWER.index(tier) + 1], ctx)     # cascade down the waterfall (NVMe is unbounded)
        self.lru[tier][blk] = None
        self.where[blk] = tier

    def _make_room(self, n: int, protect: Set[Block], ctx: Context) -> None:
        while len(self.lru["HBM"]) + n > self.cap["HBM"]:
            cand = [b for b in self.lru["HBM"] if b not in protect]
            settled = [b for b in cand if self.ready.get(b, 0.0) <= ctx.now] or cand
            victim = self.policy.hbm_victim(settled, ctx)
            del self.lru["HBM"][victim]
            self.ready.pop(victim, None)
            if victim in self.pf:
                self.pf.discard(victim)
                self.m["wasted"] += self.cfg.block_bytes
            self._insert_lower(victim, self.policy.demote_tier(victim, ctx, self._free()), ctx)

    def _fetch(self, blk: Block, now: float, prefetch: bool) -> float:
        tier = self.where[blk]
        del self.lru[tier][blk]
        done = self.backend.reserve_read(tier, self.cfg.block_bytes, now)
        self.m["busy"][tier] += self.backend.transfer_ms(tier, self.cfg.block_bytes)    # type: ignore[attr-defined]
        self.m["moved"] += self.cfg.block_bytes
        self.where[blk] = "HBM"
        self.lru["HBM"][blk] = None
        self.ready[blk] = done
        if prefetch:
            self.pf.add(blk)
            self.m["pf_bytes"] += self.cfg.block_bytes
        return done

    def _ctx(self, now: float, s: int, gap: float, hint: int | None) -> Context:
        p, pm = self.pred.make_prediction(s) if self.pred.last_pred is None else (self.pred.last_pred, self.pred.last_p)
        return Context(now=now, session=s, util={t: self.backend.utilization(t, now, foreign_only=True) for t in LOWER},
                       queue_ms={t: self.backend.queue_ms(t, now) for t in LOWER}, pred=p, pred_p_model=pm,
                       pred_accuracy=self.pred.accuracy, mean_gap_ms=gap, access_score=self.score,
                       future_hint=hint if self.policy.clairvoyant else None,
                       extra={"session_blocks": self.cfg.session_blocks, "block_bytes": self.cfg.block_bytes})

    def _missing_by_tier(self, session: int) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for b in range(self.cfg.session_blocks):
            t = self.where.get((session, b))
            if t and t != "HBM":
                out[t] = out.get(t, 0) + self.cfg.block_bytes
        return out

    # ── main loop ────────────────────────────────────────────
    def run(self) -> dict:
        c = self.cfg
        trace = make_trace(c.scenario, c.n_sessions, c.requests, c.seed)
        arr_rng = random.Random(f"arrivals|{c.scenario}|{c.seed}")
        arrival, server_free, gap_est = 0.0, 0.0, c.period_ms
        lat: List[float] = []
        stalls: List[float] = []
        first_arrival = last_end = None
        for i, s in enumerate(trace):
            gap = c.period_ms * arr_rng.uniform(0.5, 1.5)
            arrival += gap
            gap_est += (gap - gap_est) * 0.05
            start = max(arrival, server_free)
            self.pred.observe(s)
            for k in self.score:
                self.score[k] *= c.decay
            self.score[s] = self.score.get(s, 0.0) + 1.0
            hint = trace[i + 1] if i + 1 < len(trace) else None
            ctx = self._ctx(start, s, gap_est, hint)

            blocks = [(s, b) for b in range(c.session_blocks)]
            protect = set(blocks)
            new = [b for b in blocks if b not in self.where]
            miss = [b for b in blocks if b in self.where and self.where[b] != "HBM"]
            self._make_room(len(new) + len(miss), protect, ctx)
            done = start
            for b in new:                                 # first touch: the prefill output is created directly in HBM (cost not modelled)
                self.where[b] = "HBM"
                self.lru["HBM"][b] = None
            for b in miss:
                done = max(done, self._fetch(b, start, prefetch=False))
            for b in blocks:
                if self.where[b] == "HBM" and b not in miss and b not in new:
                    done = max(done, self.ready.get(b, 0.0))      # prefetched block: wait for the part not yet arrived
                    self.m["hit"] += 1
                self.pf.discard(b)
                self.lru["HBM"].move_to_end(b) if b in self.lru["HBM"] else None
            self.m["miss"] += len(miss)
            stall = max(0.0, done - start)
            end = start + stall + c.compute_ms
            server_free = end

            ctx = self._ctx(start, s, gap_est, hint)
            want = {x: self._missing_by_tier(x) for x in {ctx.pred, ctx.future_hint} if x is not None}
            q = self.policy.prefetch(ctx, want, self.est)
            if q is not None and q != s:
                todo = [(q, b) for b in range(c.session_blocks) if self.where.get((q, b)) not in (None, "HBM")]
                if todo:
                    self._make_room(len(todo), protect | {(q, b) for b in range(c.session_blocks)}, ctx)
                    for b in todo:
                        self._fetch(b, start, prefetch=True)
                    self.m["pf_issued"] += 1

            if i >= c.warmup:
                lat.append(end - arrival)
                stalls.append(stall)
                first_arrival = arrival if first_arrival is None else first_arrival
                last_end = end
        span = max(1e-9, (last_end or 1.0) - (first_arrival or 0.0))
        wasted = self.m["wasted"] + len(self.pf) * c.block_bytes
        total = self.m["hit"] + self.m["miss"]
        return {
            "scenario": c.scenario, "policy": self.policy.name, "seed": c.seed, "bg_util": dict(c.bg_util),
            "mean_latency_ms": sum(lat) / len(lat), "p50_ms": _pct(lat, 0.5), "p99_ms": _pct(lat, 0.99),
            "mean_stall_ms": sum(stalls) / len(stalls), "throughput_rps": len(lat) / span * 1000.0,
            "hbm_block_hit_rate": self.m["hit"] / total if total else 0.0,
            "bytes_moved_gib": self.m["moved"] / 2**30, "prefetch_gib": self.m["pf_bytes"] / 2**30,
            "wasted_prefetch_gib": wasted / 2**30, "prefetches_issued": self.m["pf_issued"],
            "predictor_accuracy": self.pred.accuracy,
            "link_busy_frac": {t: self.m["busy"][t] / (c.requests * c.period_ms) for t in LOWER},
        }


def run_one(cfg: Config, policy_name: str) -> dict:
    return Engine(cfg, ALL_POLICIES[policy_name]()).run()
