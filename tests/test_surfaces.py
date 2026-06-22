"""Surfaces expose correct, deterministic optima and a well-behaved fidelity model."""

from __future__ import annotations

import numpy as np
import pytest

from tuneforge.surfaces import SURFACES, make_surface
from tuneforge.surfaces.ackley import Ackley
from tuneforge.surfaces.branin import Branin
from tuneforge.surfaces.null_surface import NullSurface


def test_registry_complete():
    assert set(SURFACES) == {"branin", "ackley", "null"}


def test_unknown_surface_raises():
    with pytest.raises(ValueError):
        make_surface("nope")


def test_branin_optimum_reached_at_known_minimizer():
    s = Branin()
    # One of Branin's three global minimizers: (x1, x2) = (-pi, 12.275).
    x1_unit = (-np.pi - (-5.0)) / 15.0
    x2_unit = 12.275 / 15.0
    val = s.true_value(np.array([x1_unit, x2_unit]))
    assert val == pytest.approx(s.optimum, abs=1e-4)


def test_branin_optimum_is_a_lower_bound():
    s = Branin()
    rng = np.random.default_rng(0)
    vals = [s.true_value(rng.random(2)) for _ in range(500)]
    assert min(vals) >= s.optimum - 1e-6


def test_ackley_optimum_at_center():
    s = Ackley()
    assert s.true_value(np.array([0.5, 0.5])) == pytest.approx(0.0, abs=1e-9)
    assert s.optimum == pytest.approx(0.0)


def test_null_optimum_is_grid_min():
    s = NullSurface()
    assert s.optimum == pytest.approx(float(s._grid.min()))
    # Some cell-center achieves the optimum.
    i, j = np.unravel_index(np.argmin(s._grid), s._grid.shape)
    center = np.array([(i + 0.5) / 24, (j + 0.5) / 24])
    assert s.true_value(center) == pytest.approx(s.optimum)


def test_true_value_deterministic():
    s1, s2 = Branin(), Branin()
    x = np.array([0.3, 0.7])
    assert s1.true_value(x) == s2.true_value(x)


@pytest.mark.parametrize("name", ["branin", "ackley", "null"])
def test_full_fidelity_equals_truth_both_regimes(name):
    s = make_surface(name)
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.random(2)
        for regime in ("correlated", "decorrelated"):
            assert s.evaluate(x, 1.0, regime) == pytest.approx(s.true_value(x))


def test_decorrelated_proxy_is_noisier_than_correlated():
    s = make_surface("branin")
    rng = np.random.default_rng(2)
    pts = rng.random((200, 2))
    corr_dev, dec_dev = [], []
    for x in pts:
        t = s.true_value(x)
        corr_dev.append(s.evaluate(x, 1.0 / 9.0, "correlated") - t)
        dec_dev.append(s.evaluate(x, 1.0 / 9.0, "decorrelated") - t)
    assert np.std(dec_dev) > 3.0 * np.std(corr_dev)
