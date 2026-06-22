"""Tree-structured Parzen Estimator (Bergstra et al.), from scratch.

Model-based, single fidelity. After a random warm-up it splits the observed
configs into a "good" set (best ``gamma`` quantile) and the rest, fits a Parzen
(kernel-density) estimate to each, and proposes the candidate that maximizes the
ratio ``l(x) / g(x)`` -- points that look like the good configs and unlike the bad
ones. Each kernel uses an *adaptive* bandwidth (the distance to its nearest
neighbour), so the model exploits where good configs cluster and explores where
they are sparse. On a structured surface this concentrates evaluations near
promising regions; on the null surface the two densities are statistically
identical and the ratio carries no signal, so it degrades to random search.
"""

from __future__ import annotations

import numpy as np

from tuneforge.problem import Problem

GAMMA = 0.20          # fraction of observations treated as "good"
N_STARTUP = 8         # random warm-up evaluations before modelling
N_CANDIDATES = 96     # proposals scored per step
MIN_BW = 0.03
MAX_BW = 0.30
UNIFORM_MIX = 0.10    # background uniform density (avoids div-by-zero, adds exploration)


def _adaptive_bw(centers: np.ndarray) -> np.ndarray:
    """Per-center bandwidth = clipped distance to the nearest other center."""
    k = centers.shape[0]
    if k == 1:
        return np.array([MAX_BW])
    d2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return np.clip(nn, MIN_BW, MAX_BW)


def _density(x: np.ndarray, centers: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Parzen density at rows of ``x`` from Gaussian kernels with per-center bandwidths."""
    dim = x.shape[1]
    diff = x[:, None, :] - centers[None, :, :]          # (m, k, d)
    sq = np.sum(diff**2, axis=2) / (2.0 * bw[None, :] ** 2)
    norm = (2.0 * np.pi * bw[None, :] ** 2) ** (-0.5 * dim)
    dens = np.mean(norm * np.exp(-sq), axis=1)
    return (1.0 - UNIFORM_MIX) * dens + UNIFORM_MIX


def optimize(problem: Problem, rng: np.random.Generator) -> None:
    dim = problem.space.dim
    xs: list[np.ndarray] = []
    ys: list[float] = []

    while problem.can_afford(1.0):
        n = len(xs)
        if n < N_STARTUP:
            x = rng.random(dim)
            ys.append(problem.evaluate(x, 1.0))
            xs.append(x)
            continue

        X = np.asarray(xs)
        y = np.asarray(ys)
        order = np.argsort(y)
        n_good = max(2, int(np.ceil(GAMMA * n)))
        good = X[order[:n_good]]
        bad = X[order[n_good:]]
        if bad.shape[0] < 2:
            bad = X

        bw_g = _adaptive_bw(good)
        bw_b = _adaptive_bw(bad)

        # Propose candidates from the good density: pick a random good kernel and
        # draw from it; reserve a few uniform draws for exploration.
        pick = rng.integers(0, good.shape[0], size=N_CANDIDATES)
        cand = good[pick] + rng.normal(0.0, 1.0, size=(N_CANDIDATES, dim)) * bw_g[pick, None]
        n_explore = max(1, int(UNIFORM_MIX * N_CANDIDATES))
        cand[:n_explore] = rng.random((n_explore, dim))
        cand = np.clip(cand, 0.0, 1.0)

        scores = _density(cand, good, bw_g) / _density(cand, bad, bw_b)
        x = cand[int(np.argmax(scores))]
        ys.append(problem.evaluate(x, 1.0))
        xs.append(x)
