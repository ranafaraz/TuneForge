"""Regret metrics over a fixed budget.

The unit of account is *budget spent*, summed across every evaluation including
cheap low-fidelity looks. Regret is read only off full-fidelity (``b == 1``)
evaluations, whose values are exact -- an optimizer that never pays for a full
evaluation has no valid incumbent and therefore infinite regret. This is what lets
a single-fidelity method and a multi-fidelity method be compared on one axis: both
are charged for everything they spend, but only honest (exact) looks can lower the
curve.
"""

from __future__ import annotations

import numpy as np

from tuneforge.types import OptResult


def regret_curve(result: OptResult) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative budget vs best-so-far simple regret (step function).

    Returns two arrays of equal length: the budget spent after each trial and the
    simple regret of the best full-fidelity incumbent seen up to that point.
    """
    spent = 0.0
    best = float("inf")
    xs, ys = [], []
    for t in result.trials:
        spent += t.cost
        if t.fidelity >= 1.0 and t.value < best:
            best = t.value
        xs.append(spent)
        ys.append(best - result.optimum if np.isfinite(best) else float("inf"))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def budget_to_target(result: OptResult, target: float) -> float:
    """Budget spent when regret first drops to ``target`` or below (inf if never)."""
    xs, ys = regret_curve(result)
    hit = np.where(ys <= target)[0]
    return float(xs[hit[0]]) if hit.size else float("inf")


def auc_regret(result: OptResult, cap: float) -> float:
    """Budget-normalized area under the regret curve (anytime performance).

    Lower is better. The pre-incumbent prefix (regret == inf) is charged at ``cap``
    so that spending budget without ever producing an honest evaluation is penalized
    rather than ignored.
    """
    xs, ys = regret_curve(result)
    if xs.size == 0:
        return cap
    ys = np.minimum(np.where(np.isfinite(ys), ys, cap), cap)
    grid = np.concatenate([[0.0], xs])
    widths = np.diff(grid)
    area = float(np.sum(widths * ys))
    total = float(grid[-1]) or 1.0
    return area / total
