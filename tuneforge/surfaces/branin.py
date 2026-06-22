"""Branin-Hoo: the canonical smooth 2-D global-optimization benchmark.

Three global minima of value ``0.397887`` over the box
``x1 in [-5, 10], x2 in [0, 15]``. Smooth and gently curved -- exactly the kind of
exploitable structure a model-based optimizer should convert into sample-efficiency.
"""

from __future__ import annotations

import numpy as np

from tuneforge.surfaces.base import Surface

_A = 1.0
_B = 5.1 / (4.0 * np.pi**2)
_C = 5.0 / np.pi
_R = 6.0
_S = 10.0
_T = 1.0 / (8.0 * np.pi)
_OPT = 0.39788735772973816
_LO = np.array([-5.0, 0.0])
_HI = np.array([10.0, 15.0])


class Branin(Surface):
    name = "branin"

    def __init__(self) -> None:
        super().__init__(dim=2)
        self.finalize_spread()

    @property
    def optimum(self) -> float:
        return _OPT

    def true_value(self, x: np.ndarray) -> float:
        x1, x2 = _LO + np.asarray(x, dtype=float) * (_HI - _LO)
        return float(
            _A * (x2 - _B * x1**2 + _C * x1 - _R) ** 2
            + _S * (1.0 - _T) * np.cos(x1)
            + _S
        )
