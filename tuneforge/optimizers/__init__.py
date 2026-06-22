"""Optimizer registry. Each entry is a callable ``optimize(problem, rng) -> None``
that drives evaluations through the budgeted :class:`~tuneforge.problem.Problem`.

Two families:

* single fidelity -- ``random`` (baseline), ``tpe`` and ``gp_ei`` (model-based);
* multi fidelity  -- ``successive_halving`` and ``hyperband``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tuneforge.optimizers import gp_ei, hyperband, random_search, successive_halving, tpe
from tuneforge.problem import Problem

Optimizer = Callable[[Problem, "np.random.Generator"], None]

OPTIMIZERS: dict[str, Optimizer] = {
    "random": random_search.optimize,
    "tpe": tpe.optimize,
    "gp_ei": gp_ei.optimize,
    "successive_halving": successive_halving.optimize,
    "hyperband": hyperband.optimize,
}

# Which optimizers spend budget at fidelities below 1.0.
MULTI_FIDELITY = {"successive_halving", "hyperband"}
MODEL_BASED = {"tpe", "gp_ei"}


def make_optimizer(name: str) -> Optimizer:
    key = name.strip().lower()
    if key not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[key]


__all__ = ["OPTIMIZERS", "MULTI_FIDELITY", "MODEL_BASED", "make_optimizer", "Optimizer"]
