"""Environment-driven settings and a single deterministic RNG factory.

Every random draw in TuneForge comes from :meth:`Settings.rng`, seeded from a
fixed salt plus explicit integer offsets -- never from ``hash()`` or wall-clock
time -- so a given (optimizer, surface, seed) always reproduces bit-for-bit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from tuneforge.types import Config

# Fixed salt so seeds are stable across machines and Python versions.
SALT = 0x7_4_6_5  # "te" -- arbitrary constant, never change once published.

DEFAULT_BUDGET = 60.0
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Settings:
    optimizer: str = "tpe"
    surface: str = "branin"
    fidelity_regime: str = "correlated"
    budget: float = DEFAULT_BUDGET
    seed: int = DEFAULT_SEED
    backend: str = "numpy"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            optimizer=os.environ.get("TUNEFORGE_OPTIMIZER", "tpe").strip().lower(),
            surface=os.environ.get("TUNEFORGE_SURFACE", "branin").strip().lower(),
            fidelity_regime=os.environ.get("TUNEFORGE_FIDELITY", "correlated").strip().lower(),
            budget=float(os.environ.get("TUNEFORGE_BUDGET", DEFAULT_BUDGET)),
            seed=int(os.environ.get("TUNEFORGE_SEED", DEFAULT_SEED)),
            backend=os.environ.get("TUNEFORGE_BACKEND", "numpy").strip().lower(),
        )

    def to_config(self) -> Config:
        return Config(
            optimizer=self.optimizer,
            surface=self.surface,
            fidelity_regime=self.fidelity_regime,
            budget=self.budget,
            seed=self.seed,
            backend=self.backend,
        )

    def rng(self, *offsets: int) -> np.random.Generator:
        """A fresh Generator seeded from SALT, the run seed, and explicit offsets."""
        state = (SALT * 0x9E3779B1) ^ (int(self.seed) & 0xFFFFFFFF)
        for off in offsets:
            state = (state * 0x100000001B3) ^ (int(off) & 0xFFFFFFFFFFFFFFFF)
            state &= 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(state & 0xFFFFFFFF)


def rng_from_seed(seed: int, *offsets: int) -> np.random.Generator:
    """Stand-alone RNG factory for code paths that don't hold a Settings."""
    return Settings(seed=seed).rng(*offsets)
