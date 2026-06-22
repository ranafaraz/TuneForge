"""Gaussian process + Expected Improvement, from scratch (numpy only).

Model-based, single fidelity -- the Bayesian-optimization classic and a second,
independent witness to the sample-efficiency claim alongside TPE. An RBF-kernel GP
is fit to the standardized observations each step; the next point maximizes EI over
a random candidate pool. Like TPE it leans entirely on the smoothness prior, so on
the structureless null surface its posterior is flat and it explores like random.
"""

from __future__ import annotations

import math

import numpy as np

from tuneforge.problem import Problem

N_STARTUP = 5
N_CANDIDATES = 256
LENGTHSCALE = 0.15
NOISE = 1e-6
XI = 0.01  # exploration margin

_erf = np.vectorize(math.erf)


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


def _rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = np.sum(a[:, None, :] ** 2, axis=2) + np.sum(b[None, :, :] ** 2, axis=2)
    d2 = d2 - 2.0 * a @ b.T
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * LENGTHSCALE**2))


def _expected_improvement(
    cand: np.ndarray, X: np.ndarray, y: np.ndarray, best: float
) -> np.ndarray:
    K = _rbf(X, X) + (NOISE + 1e-9) * np.eye(X.shape[0])
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = _rbf(X, cand)                       # (n, m)
    mu = Ks.T @ alpha                        # (m,)
    v = np.linalg.solve(L, Ks)               # (n, m)
    var = 1.0 - np.sum(v**2, axis=0)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    # Minimization EI: improvement over current best (lower is better).
    z = (best - XI - mu) / sigma
    return (best - XI - mu) * _Phi(z) + sigma * _phi(z)


def optimize(problem: Problem, rng: np.random.Generator) -> None:
    dim = problem.space.dim
    xs: list[np.ndarray] = []
    ys: list[float] = []

    while problem.can_afford(1.0):
        if len(xs) < N_STARTUP:
            x = rng.random(dim)
            ys.append(problem.evaluate(x, 1.0))
            xs.append(x)
            continue

        X = np.asarray(xs)
        y_raw = np.asarray(ys)
        mu_y, sd_y = y_raw.mean(), y_raw.std() or 1.0
        y = (y_raw - mu_y) / sd_y
        best = float(y.min())

        cand = rng.random((N_CANDIDATES, dim))
        ei = _expected_improvement(cand, X, y, best)
        x = cand[int(np.argmax(ei))]
        ys.append(problem.evaluate(x, 1.0))
        xs.append(x)
