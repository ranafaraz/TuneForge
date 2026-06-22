"""Ackley: a rugged surface with one global basin under a lattice of local dips.

Global minimum ``0`` at the origin over the box ``[-5, 5]^2``. The many shallow
local minima punish naive local moves, but the broad global funnel still rewards a
model that captures coarse structure -- a harder cousin of Branin, not a needle.
"""

from __future__ import annotations

import numpy as np

from tuneforge.surfaces.base import Surface

_LO = np.array([-5.0, -5.0])
_HI = np.array([5.0, 5.0])


class Ackley(Surface):
    name = "ackley"

    def __init__(self) -> None:
        super().__init__(dim=2)
        self.finalize_spread()

    @property
    def optimum(self) -> float:
        return 0.0

    def true_value(self, x: np.ndarray) -> float:
        z = _LO + np.asarray(x, dtype=float) * (_HI - _LO)
        d = z.size
        s1 = np.sum(z**2)
        s2 = np.sum(np.cos(2.0 * np.pi * z))
        return float(
            -20.0 * np.exp(-0.2 * np.sqrt(s1 / d))
            - np.exp(s2 / d)
            + 20.0
            + np.e
        )
