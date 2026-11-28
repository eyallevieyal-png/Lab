"""Workload generator: a stream of requests, each served from the KV cache of one session (a session = SESSION_BLOCKS blocks).

The scenarios are chosen so that some are favourable to prediction and some are not. A controller that only wins on the favourable ones, or that loses on the
unfavourable ones, shows up in the benchmark: nothing here is tuned to make the adaptive policy look good.
"""
from __future__ import annotations

import random
from typing import List

SCENARIOS = ("agentic_loop", "zipf_iid", "uniform", "phase_shift")
DESCRIPTION = {
    "agentic_loop": "each session is followed by a fixed successor with probability 0.85 (tool-call chains), otherwise a random jump: a first-order predictor can learn it",
    "zipf_iid": "independent draws, Zipf(1.1) popularity: reuse exists, but the next session is not predictable beyond popularity",
    "uniform": "independent uniform draws over all sessions: no reuse structure, any prefetch is mostly wasted",
    "phase_shift": "a working set of 10 sessions, redrawn every 250 requests, uniform inside the phase: placement matters, prediction is weak",
}


def make_trace(scenario: str, n_sessions: int, n_requests: int, seed: int) -> List[int]:
    rng = random.Random(f"trace|{scenario}|{seed}")
    if scenario == "agentic_loop":
        order = list(range(n_sessions))
        rng.shuffle(order)
        succ = {order[i]: order[(i + 1) % n_sessions] for i in range(n_sessions)}
        s, out = rng.randrange(n_sessions), []
        for _ in range(n_requests):
            out.append(s)
            s = succ[s] if rng.random() < 0.85 else rng.randrange(n_sessions)
        return out
    if scenario == "zipf_iid":
        w = [1.0 / (k + 1) ** 1.1 for k in range(n_sessions)]
        ids = list(range(n_sessions))
        rng.shuffle(ids)
        return rng.choices(ids, weights=w, k=n_requests)
    if scenario == "uniform":
        return [rng.randrange(n_sessions) for _ in range(n_requests)]
    if scenario == "phase_shift":
        out, ws = [], []
        for i in range(n_requests):
            if i % 250 == 0:
                ws = rng.sample(range(n_sessions), 10)
            out.append(rng.choice(ws))
        return out
    raise ValueError(f"unknown scenario {scenario!r}")
