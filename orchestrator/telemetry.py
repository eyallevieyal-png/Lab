"""Telemetry and the causal predictor.

Nothing in this file can see the future: the predictor is updated only with requests that have already happened, and its accuracy is the running share of
past top-1 predictions that turned out right. A policy receives a `Context` snapshot; only a policy declared `clairvoyant` (the oracle upper bound) gets `future_hint`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


class MarkovPredictor:
    """First-order predictor of the next session, with a popularity fallback. Online, causal."""

    def __init__(self, alpha: float = 0.5, ewma: float = 0.05):
        self.trans: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.pop: Dict[int, int] = defaultdict(int)
        self.prev: Optional[int] = None
        self.last_pred: Optional[int] = None
        self.accuracy = 0.0          # running share of correct top-1 predictions (what a policy may use as p_hat)
        self.n = 0
        self.alpha, self.ewma = alpha, ewma

    def predict(self, cur: int) -> Tuple[Optional[int], float]:
        nxt = self.trans.get(cur)
        if nxt:
            best = max(nxt.items(), key=lambda kv: (kv[1], -kv[0]))
            return best[0], best[1] / (sum(nxt.values()) + self.alpha)
        if self.pop:
            best = max(self.pop.items(), key=lambda kv: (kv[1], -kv[0]))
            return best[0], best[1] / (sum(self.pop.values()) + self.alpha)
        return None, 0.0

    def observe(self, actual: int) -> None:
        """Called when a request arrives: score the pending prediction, then learn the transition."""
        if self.last_pred is not None:
            hit = 1.0 if self.last_pred == actual else 0.0
            self.n += 1
            self.accuracy += (hit - self.accuracy) * max(self.ewma, 1.0 / self.n)
        if self.prev is not None:
            self.trans[self.prev][actual] += 1
        self.pop[actual] += 1
        self.prev = actual
        self.last_pred = None

    def make_prediction(self, cur: int) -> Tuple[Optional[int], float]:
        p = self.predict(cur)
        self.last_pred = p[0]
        return p


@dataclass
class Context:
    """What a policy may look at when it decides."""
    now: float
    session: int                         # the request being served
    util: Dict[str, float]               # observed FOREIGN link utilisation per source tier (background load, excluding our own transfers): the u of H13
    queue_ms: Dict[str, float]           # observed queue wait per source tier
    pred: Optional[int]                  # predicted next session
    pred_p_model: float                  # the model's own probability for it
    pred_accuracy: float                 # measured running accuracy of past predictions
    mean_gap_ms: float                   # observed mean inter-arrival time
    access_score: Dict[int, float]       # exponentially decayed access count per session
    future_hint: Optional[int] = None    # ONLY for clairvoyant policies
    extra: dict = field(default_factory=dict)
