"""Hyperband -- successive halving hedged over its own aggressiveness.

Successive halving has one free knob: how many configs to start with versus how
cheaply to screen them. Too aggressive and good-but-slow-to-show configs get cut
early; too conservative and it behaves like random search. Hyperband sidesteps the
choice by sweeping a sequence of brackets from most to least aggressive, so it is
robust without tuning. It inherits successive halving's dependence on fidelity-rank
correlation: the same ``decorrelated`` ablation collapses it.
"""

from __future__ import annotations

import numpy as np

from tuneforge.optimizers.successive_halving import ETA, S_MAX, sh_bracket
from tuneforge.problem import Problem


def optimize(problem: Problem, rng: np.random.Generator) -> None:
    # Cycle brackets s = S_MAX..0 (aggressive -> conservative) until budget runs out.
    while problem.can_afford(ETA ** (-S_MAX)):
        for s in range(S_MAX, -1, -1):
            if not problem.can_afford(ETA ** (-s)):
                return
            if not sh_bracket(problem, rng, s=s):
                return
