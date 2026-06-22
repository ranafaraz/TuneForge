"""TuneForge: a black-box hyperparameter-optimizer benchmark against known optima.

Model-based search (TPE / GP-EI) buys sample-efficiency from response-surface
structure; multi-fidelity search (successive halving / Hyperband) buys
budget-efficiency from fidelity-rank correlation. Each edge is proven by an
ablation that removes exactly its source -- a structureless surface, or a
rank-decorrelated cheap proxy -- and watches the edge collapse to random search.
"""

from __future__ import annotations

# Pin BLAS to a single thread *before* numpy is imported. The benchmark is many
# tiny matrix ops (small GP Cholesky solves, KDE evaluations across a candidate
# pool); with multi-threaded BLAS the per-call thread-pool overhead dominates and
# contention can blow a few seconds up to minutes -- which would make CI unusable.
# One thread is both faster here and fully deterministic. setdefault so an explicit
# env wins.
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from tuneforge.config import Settings
from tuneforge.metrics import auc_regret, budget_to_target, regret_curve
from tuneforge.optimizers import OPTIMIZERS, make_optimizer
from tuneforge.problem import Problem
from tuneforge.runner import run, run_settings
from tuneforge.surfaces import SURFACES, make_surface
from tuneforge.types import Config, OptResult, Trial

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "Problem",
    "run",
    "run_settings",
    "make_optimizer",
    "OPTIMIZERS",
    "make_surface",
    "SURFACES",
    "regret_curve",
    "budget_to_target",
    "auc_regret",
    "Config",
    "OptResult",
    "Trial",
]
