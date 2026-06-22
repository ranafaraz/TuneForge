"""Budget accounting and regret metrics."""

from __future__ import annotations

import numpy as np

from tuneforge.metrics import auc_regret, budget_to_target, regret_curve
from tuneforge.problem import Problem
from tuneforge.runner import run
from tuneforge.surfaces import make_surface
from tuneforge.types import Config


def test_problem_charges_fidelity_as_cost():
    p = Problem(make_surface("branin"), budget=10.0, regime="correlated")
    p.evaluate(np.array([0.5, 0.5]), fidelity=0.25)
    assert p.spent == 0.25
    assert p.trials[-1].cost == 0.25
    assert p.remaining == 9.75


def test_problem_can_afford():
    p = Problem(make_surface("branin"), budget=1.0, regime="correlated")
    assert p.can_afford(1.0)
    p.evaluate(np.array([0.5, 0.5]), 1.0)
    assert not p.can_afford(1.0)


def test_regret_curve_non_increasing():
    r = run(Config("tpe", "branin", "correlated", 40.0, 0, "numpy"))
    _, ys = regret_curve(r)
    finite = ys[np.isfinite(ys)]
    assert np.all(np.diff(finite) <= 1e-9)


def test_regret_curve_budget_matches_spent():
    r = run(Config("random", "branin", "correlated", 30.0, 0, "numpy"))
    xs, _ = regret_curve(r)
    assert xs[-1] == r.spent


def test_budget_to_target_monotone_in_target():
    r = run(Config("gp_ei", "branin", "correlated", 50.0, 0, "numpy"))
    easy = budget_to_target(r, 5.0)
    hard = budget_to_target(r, 0.5)
    assert easy <= hard


def test_auc_lower_for_better_optimizer():
    # Anytime AUC is seed-noisy, so compare the mean over seeds: successive halving's
    # cheap screening gives it a clearly lower regret-vs-budget area than random.
    def mean_auc(opt):
        return np.mean([auc_regret(run(Config(opt, "branin", "correlated", 60.0, s, "numpy")),
                                   cap=30.0) for s in range(8)])
    assert mean_auc("successive_halving") < mean_auc("random")
