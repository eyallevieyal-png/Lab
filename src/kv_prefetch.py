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

import bisect
import random
from dataclasses import dataclass
from typing import List

_ONLINE_THRESHOLD: dict = {}   # memo of analytic break-even thresholds for the online policy


@dataclass
class KVConfig:
    t_x: float = 1.0            # transfer time of one KV cache (time unit)
    u_bg: float = 0.5           # background link utilisation
    gamma: float = 0.0          # predictor cost as a fraction of t_x
    lead: float = 1.5           # prefetch issued this many t_x before the need
    period: float = 6.0         # epochs are this many t_x apart (so epochs are nearly independent)
    epochs: int = 6000
    seed: int = 1
    bg_gaps: tuple | None = None   # empirical inter-arrival gaps (any unit) replacing the Poisson background; rescaled to utilisation u_bg


def _background(cfg: KVConfig, rng: random.Random) -> List[float]:
    """Arrival times of background transfers (each of size t_x): Poisson with rate u_bg / t_x."""
    if cfg.u_bg <= 0:
        return []
    times, t, horizon = [], 0.0, (cfg.epochs + 2) * cfg.period * cfg.t_x
    if cfg.bg_gaps:
        scale = (cfg.t_x / cfg.u_bg) / (sum(cfg.bg_gaps) / len(cfg.bg_gaps))   # mean gap becomes t_x / u_bg, the burst structure is kept
        i = 0
        while True:
            t += cfg.bg_gaps[i % len(cfg.bg_gaps)] * scale
            if t > horizon:
                return times
            times.append(t)
            i += 1
    rate = cfg.u_bg / cfg.t_x
    while True:
        t += rng.expovariate(rate)
        if t > horizon:
            return times
        times.append(t)


def mean_stall(cfg: KVConfig, policy: str, p: float, p_hat_fn=None) -> float:
    """Mean stall per epoch (in t_x units, predictor cost included) under `policy` in {'reactive', 'predictive', 'adaptive', 'online'}."""
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
        elif policy == "online":
            # load-aware: estimate the CURRENT background load from the arrivals seen in the last `window` transfer times (causal, nothing from the future),
            # then prefetch only if the accuracy exceeds the congestion-aware (M/D/1) break-even for that load; none exists when reactive already saturates the link
            t_dec = need - cfg.lead * t_x
            window = 30.0 * t_x
            lo_i = bisect.bisect_right(background, t_dec - window)
            hi_i = bisect.bisect_right(background, t_dec)
            u_now = min(0.99, (hi_i - lo_i) * t_x / window)
            key = (round(u_now, 2), cfg.gamma, cfg.lead, cfg.period)
            if key not in _ONLINE_THRESHOLD:
                _ONLINE_THRESHOLD[key] = analytic_breakeven(key[0], cfg.gamma, cfg.lead, cfg.period)
            thr = _ONLINE_THRESHOLD[key]
            use_pred = thr is not None and p >= thr
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


# ── congestion-aware analytic threshold (M/D/1 approximation) ───────────────────────────────────────────────

def md1_wait(rho: float) -> float:
    """Mean queueing wait of an M/D/1 link in units of one transfer time (Pollaczek-Khinchine), infinite at saturation."""
    return float("inf") if rho >= 1 else rho / (2 * (1 - rho))


def analytic_stall(u: float, p: float, gamma: float, lead: float, period: float, policy: str) -> float:
    """Mean stall per epoch (transfer times) with the link treated as an M/D/1 queue whose load includes this stream and its wasted prefetches."""
    if policy == "reactive":
        return 1 + md1_wait(u + 1 / period)
    rho = u + (2 - p) / period            # one transfer per epoch plus one wasted transfer with probability 1 - p
    w = md1_wait(rho)
    if w == float("inf"):
        return float("inf")
    return p * max(0.0, w + 1 - lead) + (1 - p) * (1 + w) + gamma


def analytic_breakeven(u: float, gamma: float, lead: float = 1.5, period: float = 6.0, tol: float = 1e-4) -> float | None:
    """Smallest accuracy at which the analytic predictive stall does not exceed the reactive one; None if no accuracy up to 1 does."""
    base = analytic_stall(u, 0.0, gamma, lead, period, "reactive")
    if base == float("inf"):          # the reactive policy alone already saturates the link: no meaningful break-even (Lean: Lab.no_stable_predictor)
        return None
    f = lambda p: analytic_stall(u, p, gamma, lead, period, "predictive") - base  # noqa: E731
    if f(1.0) > 0:
        return None
    if f(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def stability_floor(u: float, period: float) -> float:
    """Accuracy below which the always-prefetch policy saturates the link: p > 2 - period * (1 - u) (Lean: Lab.stable_iff)."""
    return 2 - period * (1 - u)
