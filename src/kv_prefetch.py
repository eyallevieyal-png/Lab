"""KV-cache prefetch over one host link: a queueing simulation that tests the closed-form break-even accuracy of H13.

One host link (single FIFO server, transfer time T_x per KV cache) is shared by this stream and by background traffic that arrives as a Poisson process and
keeps the link busy a fraction u_bg of the time. At each epoch k one sequence is needed at time t_k = k * period.

  reactive   : at t_k the KV cache is demanded; stall = (completion - t_k), which includes queueing behind background traffic.
  predictive : at t_k - lead the predictor's guess is prefetched. With probability p it is the right sequence: the stall is whatever part of the
               transfer is not finished by t_k. Otherwise the prefetch was wasted: it occupies the link, and the real sequence is demanded at t_k.
               Every predictive epoch also pays a predictor cost gamma * T_x.
  adaptive   : predictive only when the running estimate of the net benefit is positive, i.e. p_hat > (u_hat + gamma) / (1 + u_hat) (the Lean threshold,
               `Lab.prefetch_threshold_iff`), otherwise reactive.

Closed form being tested (formula of formal/LabFormal/Orchestration.lean): p* = (u + gamma) / (1 + u), which assumes a wrong prefetch delays the demand
fetch with probability u and a right one hides the whole transfer. The simulation does not assume that: queueing, partial hiding and the lead time decide.
Common random numbers are used for all policies, so differences between policies are not sampling noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class KVConfig:
    t_x: float = 1.0            # transfer time of one KV cache (time unit)
    u_bg: float = 0.5           # background link utilisation
    gamma: float = 0.0          # predictor cost as a fraction of t_x
    lead: float = 1.5           # prefetch issued this many t_x before the need
    period: float = 6.0         # epochs are this many t_x apart (so epochs are nearly independent)
    epochs: int = 6000
    seed: int = 1


def _background(cfg: KVConfig, rng: random.Random) -> List[float]:
    """Arrival times of background transfers (each of size t_x): Poisson with rate u_bg / t_x."""
    if cfg.u_bg <= 0:
        return []
    times, t, horizon = [], 0.0, (cfg.epochs + 2) * cfg.period * cfg.t_x
    rate = cfg.u_bg / cfg.t_x
    while True:
        t += rng.expovariate(rate)
        if t > horizon:
            return times
        times.append(t)


def mean_stall(cfg: KVConfig, policy: str, p: float, p_hat_fn=None) -> float:
    """Mean stall per epoch (in t_x units, predictor cost included) under `policy` in {'reactive', 'predictive', 'adaptive'}."""
    rng = random.Random(f"kv|{cfg.seed}|{cfg.u_bg}")          # same background and same draws for every policy
    background = _background(cfg, rng)
    draws = [rng.random() for _ in range(cfg.epochs)]          # prediction k is right iff draws[k] < p
    period, t_x = cfg.period * cfg.t_x, cfg.t_x

    # decide, per epoch, which transfers this policy issues
    reqs = [(t, "bg", -1) for t in background]
    decisions = []
    u_hat, p_hat = cfg.u_bg, p                                   # running estimates, initialised at the true values (a calibrated controller)
    for k in range(cfg.epochs):
        need = (k + 1) * period
        if policy == "reactive":
            use_pred = False
        elif policy == "predictive":
            use_pred = True
        else:  # adaptive: act on the prediction only when the closed-form net benefit is positive
            use_pred = p_hat > (u_hat + cfg.gamma) / (1 + u_hat)
        right = use_pred and draws[k] < p
        decisions.append((use_pred, right))
        if use_pred:
            reqs.append((need - cfg.lead * t_x, "pf", k))
        if not right:
            reqs.append((need, "demand", k))
    reqs.sort(key=lambda r: (r[0], r[1]))

    done = {}   # (kind, k) -> completion time; FIFO single server
    free = 0.0
    for arr, kind, k in reqs:
        start = max(arr, free)
        free = start + t_x
        done[(kind, k)] = free

    total = 0.0
    for k, (use_pred, right) in enumerate(decisions):
        need = (k + 1) * period
        stall = max(0.0, done[("pf", k)] - need) if right else done[("demand", k)] - need
        total += stall / t_x + (cfg.gamma if use_pred else 0.0)
    return total / cfg.epochs


def breakeven_accuracy(cfg: KVConfig, tol: float = 1e-3) -> float | None:
    """Smallest accuracy p at which predictive no longer stalls more than reactive (bisection with common random numbers).
    None if even a perfect predictor does not pay off (predictive worse than reactive at p = 1)."""
    base = mean_stall(cfg, "reactive", 0.0)
    f = lambda p: mean_stall(cfg, "predictive", p) - base  # noqa: E731  (decreasing in p)
    if f(1.0) > 0:
        return None
    if f(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def closed_form(u: float, gamma: float) -> float:
    return (u + gamma) / (1 + u)
