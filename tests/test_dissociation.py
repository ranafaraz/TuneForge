"""The headline claims, in miniature (few seeds): both effects dissociate.

The full-strength, tightly-thresholded version lives in ``evals/gate.py`` and runs
as its own CI step. These tests guard the *direction* of each effect cheaply so a
regression in any optimizer trips ``pytest`` immediately.
"""

from __future__ import annotations

import numpy as np

from tuneforge.metrics import auc_regret
from tuneforge.runner import run
from tuneforge.types import Config

SEEDS = 8


def _mean_regret(opt, surf, regime="correlated"):
    return float(np.mean([run(Config(opt, surf, regime, 60.0, s, "numpy")).simple_regret
                          for s in range(SEEDS)]))


def _mean_auc(opt, surf, regime, cap):
    return float(np.mean([auc_regret(run(Config(opt, surf, regime, 60.0, s, "numpy")), cap=cap)
                          for s in range(SEEDS)]))


def test_model_based_beats_random_on_structured_surface():
    base = _mean_regret("random", "branin")
    assert _mean_regret("tpe", "branin") < base
    assert _mean_regret("gp_ei", "branin") < base


def test_model_based_advantage_collapses_on_null():
    base = _mean_regret("random", "null")
    # No meaningful speedup once structure is removed.
    assert _mean_regret("tpe", "null") >= base / 1.5
    assert _mean_regret("gp_ei", "null") >= base / 1.5


def test_model_based_advantage_is_far_larger_on_structure_than_null():
    b_adv = _mean_regret("random", "branin") / _mean_regret("gp_ei", "branin")
    n_adv = _mean_regret("random", "null") / _mean_regret("gp_ei", "null")
    assert b_adv > 2.0 * n_adv


def test_multifidelity_beats_random_under_correlated_proxy():
    base = _mean_auc("random", "branin", "correlated", 30.0)
    assert _mean_auc("successive_halving", "branin", "correlated", 30.0) < base
    assert _mean_auc("hyperband", "branin", "correlated", 30.0) < base


def test_multifidelity_advantage_collapses_under_decorrelated_proxy():
    base = _mean_auc("random", "branin", "decorrelated", 30.0)
    # With a useless proxy the early eliminations are random -> worse than random.
    assert _mean_auc("successive_halving", "branin", "decorrelated", 30.0) > base
    assert _mean_auc("hyperband", "branin", "decorrelated", 30.0) > base
