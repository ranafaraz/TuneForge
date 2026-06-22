"""The budgeted black box handed to every optimizer.

An optimizer never touches a :class:`Surface` directly: it asks the ``Problem`` to
evaluate a unit-cube point at a chosen fidelity. The Problem records the trial,
charges the budget, and returns the (possibly biased, low-fidelity) observation.
This keeps budget accounting in exactly one place and guarantees every optimizer is
charged identically.
"""

from __future__ import annotations

import numpy as np

from tuneforge.surfaces.base import Surface
from tuneforge.types import Trial


class Problem:
    def __init__(self, surface: Surface, budget: float, regime: str = "correlated") -> None:
        self.surface = surface
        self.space = surface.space
        self.regime = regime
        self.budget = float(budget)
        self.trials: list[Trial] = []

    @property
    def spent(self) -> float:
        return float(sum(t.cost for t in self.trials))

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    def evaluate(self, x: np.ndarray, fidelity: float = 1.0) -> float:
        x = self.space.clip(np.asarray(x, dtype=float))
        fidelity = float(np.clip(fidelity, 1e-6, 1.0))
        value = self.surface.evaluate(x, fidelity, self.regime)
        self.trials.append(
            Trial(x=self.space.as_tuple(x), fidelity=fidelity, value=value, cost=fidelity)
        )
        return value

    def can_afford(self, cost: float) -> bool:
        # Small epsilon so floating-point budget arithmetic doesn't strand a final eval.
        return self.remaining >= cost - 1e-9
