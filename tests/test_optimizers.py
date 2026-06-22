"""Every optimizer stays within budget, produces a valid incumbent, and is
deterministic given a seed."""

from __future__ import annotations

import numpy as np
import pytest

from tuneforge.optimizers import MULTI_FIDELITY, OPTIMIZERS, make_optimizer
from tuneforge.problem import Problem
from tuneforge.runner import run
from tuneforge.surfaces import make_surface
from tuneforge.types import Config

ALL = list(OPTIMIZERS)


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        make_optimizer("nope")


@pytest.mark.parametrize("name", ALL)
def test_respects_budget(name):
    surf = make_surface("branin")
    problem = Problem(surf, budget=40.0, regime="correlated")
    make_optimizer(name)(problem, np.random.default_rng(0))
    assert problem.spent <= 40.0 + 1e-6


@pytest.mark.parametrize("name", ALL)
def test_produces_full_fidelity_incumbent(name):
    r = run(Config(name, "branin", "correlated", 40.0, 0, "numpy"))
    assert r.incumbent_x is not None
    assert np.isfinite(r.incumbent_value)
    assert any(t.fidelity >= 1.0 for t in r.trials)
    assert r.simple_regret >= -1e-9


@pytest.mark.parametrize("name", ALL)
def test_deterministic(name):
    a = run(Config(name, "branin", "correlated", 40.0, 1, "numpy"))
    b = run(Config(name, "branin", "correlated", 40.0, 1, "numpy"))
    assert [t.value for t in a.trials] == [t.value for t in b.trials]
    assert a.simple_regret == b.simple_regret


def test_random_does_one_full_eval_per_unit_budget():
    r = run(Config("random", "branin", "correlated", 25.0, 0, "numpy"))
    assert r.n_trials == 25
    assert all(t.fidelity == 1.0 for t in r.trials)


@pytest.mark.parametrize("name", sorted(MULTI_FIDELITY))
def test_multifidelity_uses_cheap_looks(name):
    r = run(Config(name, "branin", "correlated", 60.0, 0, "numpy"))
    assert any(t.fidelity < 1.0 for t in r.trials)  # spent budget below full fidelity
    assert any(t.fidelity >= 1.0 for t in r.trials)  # but still finished a bracket
