"""Successive halving -- the multi-fidelity workhorse.

Sample many configs, evaluate them all cheaply at a low fidelity, throw away the
worst ``1 - 1/eta`` fraction, then re-evaluate the survivors at a higher fidelity,
and repeat until a single config is checked at full fidelity. The bet is that the
cheap low-fidelity ranking predicts the expensive one. When it does (the
``correlated`` regime) the method reaches good full-fidelity configs for far less
budget than evaluating everything at full fidelity; when it doesn't
(``decorrelated``) the early eliminations are essentially random and the budget is
wasted.

This module also exports :func:`sh_bracket`, the single-bracket primitive Hyperband
composes over.
"""

from __future__ import annotations

import numpy as np

from tuneforge.problem import Problem

ETA = 3
S_MAX = 2  # max rungs: fidelities 1/9 -> 1/3 -> 1 (R=1, r_min=1/9)


def sh_bracket(problem: Problem, rng: np.random.Generator, s: int) -> bool:
    """Run one successive-halving bracket. Returns False if budget ran out mid-bracket."""
    dim = problem.space.dim
    n = int(np.ceil((S_MAX + 1) / (s + 1) * ETA**s))
    configs = rng.random((max(n, 1), dim))

    for i in range(s + 1):
        ni = max(1, int(np.floor(n * ETA ** (-i))))
        fidelity = min(1.0, float(ETA ** (i - s)))
        configs = configs[:ni]
        vals = np.empty(configs.shape[0])
        for j, c in enumerate(configs):
            if not problem.can_afford(fidelity):
                return False
            vals[j] = problem.evaluate(c, fidelity)
        keep = max(1, int(np.floor(configs.shape[0] / ETA)))
        configs = configs[np.argsort(vals)[:keep]]
    return True


def optimize(problem: Problem, rng: np.random.Generator) -> None:
    # Repeat the most aggressive bracket until the budget is exhausted.
    while problem.can_afford(ETA ** (-S_MAX)):
        if not sh_bracket(problem, rng, s=S_MAX):
            break
