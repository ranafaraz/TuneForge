"""Null surface: a deliberately *structureless* control.

The unit square is diced into a grid of cells, each given an independent
deterministic pseudo-random value. Neighbouring cells are unrelated, so there is
no gradient, no basin, nothing for a smoothness prior or a density model to
exploit. This is the ablation for the *sample-efficiency* claim: on a structured
surface a model-based optimizer should beat random search; here -- with the
structure removed but everything else identical -- it must fall back to random.

The global optimum is simply the lowest cell value (computed exactly at build
time), so regret is still well defined and bounded.
"""

from __future__ import annotations

import numpy as np

from tuneforge.config import SALT, rng_from_seed
from tuneforge.surfaces.base import Surface

_GRID = 24  # 24 x 24 = 576 independent cells


class NullSurface(Surface):
    name = "null"

    def __init__(self) -> None:
        super().__init__(dim=2)
        # An independent value per cell -- white noise over the square.
        grng = rng_from_seed(SALT, 0x4E0117)
        self._grid = grng.random((_GRID, _GRID))
        self._opt = float(self._grid.min())
        self.finalize_spread()

    @property
    def optimum(self) -> float:
        return self._opt

    def _cell(self, x: np.ndarray) -> tuple[int, int]:
        idx = np.clip((np.asarray(x) * _GRID).astype(int), 0, _GRID - 1)
        return int(idx[0]), int(idx[1])

    def true_value(self, x: np.ndarray) -> float:
        i, j = self._cell(x)
        return float(self._grid[i, j])
