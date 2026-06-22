"""The search space optimizers operate in: the unit cube ``[0, 1]^dim``.

Optimizers only ever see unit-cube points; each surface owns the mapping from the
unit cube to its native bounds. This keeps every optimizer surface-agnostic and
makes distance-based models (GP, TPE KDE) well scaled by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchSpace:
    dim: int

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """``n`` uniform points in the unit cube, shape ``(n, dim)``."""
        return rng.random((n, self.dim))

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def as_tuple(x: np.ndarray) -> tuple[float, ...]:
        return tuple(float(v) for v in np.ravel(x))
