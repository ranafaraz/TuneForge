"""Core dataclasses shared across TuneForge.

A *surface* is the black box being optimized (an analytic function with a known
global minimum). An *optimizer* spends a budget of evaluations on it and returns
an :class:`OptResult` -- a full trace of trials plus the recommended incumbent.
Everything downstream (metrics, eval harness, gate) reads these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Trial:
    """A single black-box evaluation.

    ``x`` is the point in the (unit-cube) search space, ``fidelity`` is the
    fraction of full budget spent on it (1.0 == exact), ``value`` is the observed
    objective (lower is better), and ``cost`` is the budget it consumed (== fidelity).
    """

    x: tuple[float, ...]
    fidelity: float
    value: float
    cost: float


@dataclass
class OptResult:
    """The outcome of running one optimizer on one surface for one seed."""

    optimizer: str
    surface: str
    seed: int
    budget: float
    trials: list[Trial] = field(default_factory=list)
    # Recommended incumbent: best config evaluated at full fidelity (b == 1.0).
    incumbent_x: tuple[float, ...] | None = None
    incumbent_value: float = float("inf")
    optimum: float = 0.0

    @property
    def spent(self) -> float:
        """Total budget consumed across every trial (low-fidelity looks included)."""
        return float(sum(t.cost for t in self.trials))

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def simple_regret(self) -> float:
        """Gap between the recommended incumbent's true value and the optimum."""
        return float(self.incumbent_value - self.optimum)


@dataclass(frozen=True)
class Config:
    """A fully resolved experiment specification (what to run, deterministically)."""

    optimizer: str
    surface: str
    fidelity_regime: str
    budget: float
    seed: int
    backend: str


def as_array(x: tuple[float, ...]) -> np.ndarray:
    return np.asarray(x, dtype=float)
