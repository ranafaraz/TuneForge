"""Uniform random search -- the baseline every other optimizer must beat.

Single fidelity: every config is evaluated exactly once at full budget. Despite its
simplicity it is a famously strong baseline, which is exactly why it is the honest
reference point for both ablations.
"""

from __future__ import annotations

import numpy as np

from tuneforge.problem import Problem


def optimize(problem: Problem, rng: np.random.Generator) -> None:
    while problem.can_afford(1.0):
        x = rng.random(problem.space.dim)
        problem.evaluate(x, fidelity=1.0)
