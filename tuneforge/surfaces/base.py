"""Response-surface base class: a black box with a *known* global minimum.

A surface exposes two things an optimizer can call:

* :meth:`true_value` -- the exact objective at full fidelity (lower is better).
* :meth:`evaluate`   -- the objective observed at a given *fidelity* ``b in (0, 1]``.

At ``b == 1`` the two coincide exactly. For ``b < 1`` the observation is a cheaper,
biased proxy whose informativeness is governed by the *fidelity regime*:

* ``correlated``   -- a small smooth bias; the cheap proxy's *ranking* of configs
  tracks the truth, so screening at low fidelity is worthwhile.
* ``decorrelated`` -- a large rough bias that dominates at low fidelity, so the
  proxy's ranking is independent of the truth and screening is a waste.

That single knob is the independent variable behind the multi-fidelity ablation:
Hyperband's edge should survive ``correlated`` and collapse under ``decorrelated``.
Crucially the regime never changes the ``b == 1`` value, so it cannot move the final
regret directly -- only the *path* (which configs get promoted) changes.
"""

from __future__ import annotations

import numpy as np

from tuneforge.config import SALT, rng_from_seed
from tuneforge.spaces import SearchSpace

# Amplitudes of the two bias fields, expressed as multiples of the surface's own
# spread (std of the true objective). Small + smooth keeps rank under `correlated`;
# large + rough destroys rank under `decorrelated`.
CORR_AMP = 0.15
DECORR_AMP = 6.0
_FIELD_GRID = 41  # resolution of the rough (decorrelating) noise field per axis


class Surface:
    name: str = "base"

    def __init__(self, dim: int) -> None:
        self.space = SearchSpace(dim)
        self.dim = dim
        # Placeholder spread; subclasses call finalize_spread() once true_value works.
        self._t_std = 1.0
        # Deterministic field parameters (fixed across regimes/seeds via SALT).
        frng = rng_from_seed(SALT, 0xF1E1D)
        self._smooth_freq = frng.integers(1, 3, size=(3, dim)).astype(float)
        self._smooth_phase = frng.uniform(0, 2 * np.pi, size=3)
        self._smooth_amp = frng.uniform(0.5, 1.0, size=3)
        # Rough field: a fine grid of independent values in [-1, 1].
        self._noise_grid = frng.uniform(-1.0, 1.0, size=(_FIELD_GRID,) * dim)

    # --- to be provided by concrete surfaces -------------------------------
    @property
    def optimum(self) -> float:
        raise NotImplementedError

    def true_value(self, x: np.ndarray) -> float:
        """Exact objective at unit-cube point ``x`` (shape ``(dim,)``)."""
        raise NotImplementedError

    # --- shared machinery ---------------------------------------------------
    def _smooth_field(self, x: np.ndarray) -> float:
        """Low-frequency, bounded perturbation in roughly ``[-1, 1]``."""
        phases = 2 * np.pi * (self._smooth_freq @ x) + self._smooth_phase
        return float(np.sum(self._smooth_amp * np.sin(phases)) / np.sum(self._smooth_amp))

    def _noise_field(self, x: np.ndarray) -> float:
        """High-frequency, rank-destroying perturbation in ``[-1, 1]``."""
        idx = np.clip((np.asarray(x) * _FIELD_GRID).astype(int), 0, _FIELD_GRID - 1)
        return float(self._noise_grid[tuple(idx)])

    def evaluate(self, x: np.ndarray, fidelity: float = 1.0, regime: str = "correlated") -> float:
        x = np.asarray(x, dtype=float)
        t = self.true_value(x)
        b = float(fidelity)
        if b >= 1.0:
            return t
        if regime == "decorrelated":
            return t + (1.0 - b) * DECORR_AMP * self._t_std * self._noise_field(x)
        # default: correlated
        return t + (1.0 - b) * CORR_AMP * self._t_std * self._smooth_field(x)

    def finalize_spread(self) -> None:
        """Compute the true-objective spread once the subclass is fully built."""
        g = 21
        lin = np.linspace(0.0, 1.0, g)
        mesh = np.stack([m.ravel() for m in np.meshgrid(*([lin] * self.dim))], axis=1)
        vals = np.array([self.true_value(p) for p in mesh])
        self._t_std = float(vals.std()) or 1.0
