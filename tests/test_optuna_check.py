"""Optional Optuna cross-check. Skips cleanly when the [optuna] extra is absent,
mirroring the offline-first contract (CI's default path never needs it)."""

from __future__ import annotations

import numpy as np
import pytest

from tuneforge.runner import run
from tuneforge.types import Config

optuna = pytest.importorskip("optuna")


def test_optuna_random_matches_budget():
    r = run(Config("random", "branin", "correlated", 30.0, 0, "optuna"))
    assert r.n_trials == 30
    assert np.isfinite(r.incumbent_value)
    assert r.simple_regret >= -1e-9


def test_optuna_tpe_beats_random_on_branin():
    tpe = run(Config("tpe", "branin", "correlated", 60.0, 0, "optuna"))
    rnd = run(Config("random", "branin", "correlated", 60.0, 0, "optuna"))
    # Independent implementation reproduces the headline ranking.
    assert tpe.simple_regret < rnd.simple_regret
