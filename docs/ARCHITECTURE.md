# Architecture

TuneForge is a benchmark, not a tuning library. Its product is a *measurement*: how
much budget each optimizer needs to approach a known optimum, and which property of the
problem each optimizer's advantage actually depends on. The design exists to make that
measurement honest, deterministic, and reproducible offline.

## The pieces

```
tuneforge/
  types.py        Trial, OptResult, Config  (the data that flows between layers)
  config.py       Settings + a single deterministic RNG factory (SALT + offsets)
  spaces.py       SearchSpace: the unit cube [0,1]^d every optimizer works in
  problem.py      Problem: the budgeted black box; charges cost == fidelity
  surfaces/       Branin, Ackley, Null + the shared fidelity model (base.py)
  optimizers/     random, tpe, gp_ei, successive_halving, hyperband + registry
  runner.py       run(Config) -> OptResult; picks the full-fidelity incumbent
  optuna_check.py optional Optuna cross-check backend ([optuna] extra)
  metrics.py      simple regret, regret-vs-budget curve, AUC, budget-to-target
  cli.py          tuneforge optimize | compare | surfaces | eval
evals/
  metrics.py      aggregate across seeds -> "advantage over random" ratios
  harness.py      run the full grid, write RESULTS.md
  gate.py         assert the two-effect dissociation; non-zero exit on regression
```

## One black box, one budget axis

Every optimizer talks only to a `Problem`. It asks the Problem to evaluate a unit-cube
point at a chosen fidelity `b ∈ (0, 1]`; the Problem records a `Trial`, **charges budget
equal to `b`**, and returns the observation. This is the crux of a fair comparison
between single- and multi-fidelity methods:

- A single-fidelity method (random, TPE, GP-EI) always pays `b = 1` and gets an exact value.
- A multi-fidelity method (successive halving, Hyperband) pays `b < 1` for cheap, biased
  looks and only occasionally pays `b = 1`.

Both are charged for *everything they spend*, and **regret is read only off `b = 1`
evaluations** (which are exact). So an optimizer cannot "win" by hoarding cheap looks: to
lower its regret curve it must eventually spend on an honest evaluation. The regret-vs-
budget AUC then compares all five optimizers on a single axis — total budget spent.

## Why the surfaces are shaped the way they are

| surface | role | what it tests |
|---|---|---|
| **branin** | smooth, 3 global minima at 0.397887 | structure a model can exploit |
| **ackley** | one global basin under many local dips | harder structure, not a needle |
| **null** | grid of independent random cells | the **structure-removed control** |

The **fidelity model** (in `surfaces/base.py`) is shared by all surfaces. At `b < 1` it
adds either a small smooth bias (`correlated`: rank mostly preserved, so cheap screening
is informative) or a large rough bias (`decorrelated`: rank destroyed, so screening is
noise). It is **exact at `b = 1`**, which is what makes the ablation clean: changing the
regime cannot move the final regret directly — it can only change which configs a
multi-fidelity optimizer chooses to promote.

## Two independent ablations

- **Structure ablation** (sample-efficiency): run model-based optimizers on `null`. Their
  advantage over random collapses, isolating "the win came from exploitable structure."
- **Fidelity-correlation ablation** (budget-efficiency): run multi-fidelity optimizers
  with the `decorrelated` proxy. Their advantage collapses, isolating "the win came from
  the cheap proxy ranking configs like the expensive one."

Crucially the two ablations are orthogonal: removing structure leaves the multi-fidelity
result alone, and decorrelating the proxy leaves the sample-efficiency result alone. That
orthogonality is what lets the two effects be reported as *separate* findings.

## Determinism

All randomness comes from `Settings.rng(*offsets)`, seeded from a fixed `SALT` plus
integer offsets — never `hash()` or the clock. `numpy.random.default_rng` is stable
across platforms and Python versions, so the eval numbers are identical on every CI
runner, which is what allows the gate to use tight thresholds without flaking.

BLAS is pinned to one thread before numpy imports (see `tuneforge/__init__.py`): the
benchmark is thousands of tiny matrix ops (small GP Cholesky solves, KDE evaluations) where
multi-threaded BLAS is both slower and non-deterministic.
