"""Deterministic RNG, env parsing, and the unit-cube search space."""

from __future__ import annotations

import numpy as np

from tuneforge.config import Settings, rng_from_seed
from tuneforge.spaces import SearchSpace


def test_rng_same_seed_same_stream():
    a = Settings(seed=3).rng(7, 11).random(5)
    b = Settings(seed=3).rng(7, 11).random(5)
    assert np.array_equal(a, b)


def test_rng_different_seed_differs():
    a = Settings(seed=3).rng(7).random(5)
    b = Settings(seed=4).rng(7).random(5)
    assert not np.array_equal(a, b)


def test_rng_different_offsets_differ():
    a = Settings(seed=3).rng(1).random(5)
    b = Settings(seed=3).rng(2).random(5)
    assert not np.array_equal(a, b)


def test_rng_from_seed_helper_matches_settings():
    a = rng_from_seed(9, 2, 3).random(4)
    b = Settings(seed=9).rng(2, 3).random(4)
    assert np.array_equal(a, b)


def test_from_env_defaults(monkeypatch):
    for k in ("TUNEFORGE_OPTIMIZER", "TUNEFORGE_SURFACE", "TUNEFORGE_FIDELITY",
              "TUNEFORGE_BUDGET", "TUNEFORGE_SEED", "TUNEFORGE_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert (s.optimizer, s.surface, s.fidelity_regime) == ("tpe", "branin", "correlated")
    assert s.backend == "numpy"


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("TUNEFORGE_OPTIMIZER", "hyperband")
    monkeypatch.setenv("TUNEFORGE_SURFACE", "null")
    monkeypatch.setenv("TUNEFORGE_BUDGET", "30")
    s = Settings.from_env()
    assert s.optimizer == "hyperband" and s.surface == "null" and s.budget == 30.0


def test_space_sample_shape_and_bounds():
    sp = SearchSpace(dim=2)
    x = sp.sample(np.random.default_rng(0), n=10)
    assert x.shape == (10, 2)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_space_clip():
    sp = SearchSpace(dim=3)
    clipped = sp.clip(np.array([-0.2, 0.5, 1.4]))
    assert np.array_equal(clipped, np.array([0.0, 0.5, 1.0]))
