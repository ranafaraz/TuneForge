"""Aggregate an optimizer's behaviour across seeds.

Two summary numbers carry the whole thesis:

* the *sample-efficiency* of a single-fidelity optimizer = how much lower its mean
  simple regret is than random search at equal budget;
* the *budget-efficiency* of a multi-fidelity optimizer = how much lower its mean
  regret-vs-budget AUC is than random search.

Each is reported as an "advantage" ratio over random (>1 means better than random),
so the ablation reads directly: the advantage is large on the structured/correlated
setting and falls to ~1 (or below) once its source is removed.
"""

from __future__ import annotations

import numpy as np

from tuneforge.metrics import auc_regret
from tuneforge.runner import run
from tuneforge.types import Config


def regrets(optimizer: str, surface: str, regime: str, budget: float, seeds: int) -> np.ndarray:
    return np.array(
        [run(Config(optimizer, surface, regime, budget, s, "numpy")).simple_regret
         for s in range(seeds)]
    )


def aucs(optimizer: str, surface: str, regime: str, budget: float, seeds: int,
         cap: float) -> np.ndarray:
    return np.array(
        [auc_regret(run(Config(optimizer, surface, regime, budget, s, "numpy")), cap=cap)
         for s in range(seeds)]
    )


def mean_regret(optimizer: str, surface: str, budget: float, seeds: int,
                regime: str = "correlated") -> float:
    return float(regrets(optimizer, surface, regime, budget, seeds).mean())


def mean_auc(optimizer: str, surface: str, regime: str, budget: float, seeds: int,
             cap: float) -> float:
    return float(aucs(optimizer, surface, regime, budget, seeds, cap).mean())


def advantage(baseline: float, contender: float) -> float:
    """Ratio baseline/contender (>1 means the contender beats random)."""
    return float(baseline / contender) if contender > 0 else float("inf")
