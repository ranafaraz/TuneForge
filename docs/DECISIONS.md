# Decisions & trade-offs

What I built, and the calls I made building it.

## Scope: a benchmark with ground truth, not a tuning toolkit

There are excellent HPO libraries already (Optuna, Hyperopt, Ax). Reimplementing one adds
nothing. The gap worth filling is **explanatory**: optimizers are usually compared by a
single leaderboard number on opaque tasks, which tells you *that* one wins but not *why*,
and gives no way to falsify the explanation. TuneForge fixes the objective to an analytic
function with a known optimum, so regret is exact, and then attacks the *why* with
ablations. The deliverable is a dissociation of two effects, each proven by collapsing it.

This is deliberately a different lane from the rest of the portfolio: not LLM/RAG, not
finance back-testing, and not [MLForge](https://github.com/ranafaraz/MLForge), which owns
*model selection* and the leakage/selection-bias of a CV score. TuneForge owns the
*optimizers themselves* and the sample- vs budget-efficiency split.

## Why these two effects, and why ablate

A leaderboard can be gamed by tuning the benchmark. A *dissociation* can't be: the claim
"model-based search wins because of structure" is only credible if removing the structure
removes the win. So every headline number is paired with a control where it must vanish:

- model-based vs random on `null` (structure removed),
- multi-fidelity vs random with the `decorrelated` proxy (fidelity correlation removed).

The gate asserts the *shape* — win here, collapse there — not just a single threshold, so
a regression that, say, accidentally lets the GP cheat on `null` fails CI loudly.

## Regret only from full-fidelity evaluations

The hardest design question was how to compare a single-fidelity method against a
multi-fidelity one fairly. The answer: charge budget for *everything* (cost = fidelity),
but let **only exact (`b = 1`) evaluations lower the regret curve**. This makes the
incentive honest — cheap looks are worth paying for *only* insofar as they lead you to
spend a full evaluation in a good place — and it means an optimizer that never evaluates at
full fidelity correctly has infinite regret. Hyperband's design guarantees its top rung is
`b = 1`, so it always produces a valid incumbent.

## The fidelity model is exact at b = 1 — on purpose

It would have been easy to let low fidelity be "noisy everywhere including b = 1." I chose
to make `b = 1` exact and identical across regimes. The payoff: the `correlated` vs
`decorrelated` regimes **cannot change the final score directly** — they only change which
configs get promoted through the rungs. That isolates the multi-fidelity advantage to
exactly one cause (proxy rank correlation) and makes the ablation unimpeachable.

## TPE and GP-EI from scratch

Two model-based optimizers, not one, so the sample-efficiency claim has two independent
witnesses with very different machinery (kernel-density ratio vs Gaussian-process
posterior). Both lean entirely on a smoothness/locality prior, so both must — and do —
collapse on `null`. Writing them on numpy keeps the core dependency-free and CI offline; an
early bug where TPE used a single global bandwidth (~0.38 of the unit cube) made it *worse*
than random, fixed by switching to adaptive per-point bandwidths (distance to nearest
neighbour), the standard Bergstra trick.

## Calibration: budget 60, η = 3, 16 seeds

- **Budget 60** is large enough that model-based search clearly separates from random on
  branin, but small enough that the run is seconds and the comparison is in the
  sample-limited regime HPO actually cares about.
- **η = 3** gives a clean three-rung ladder (1/9 → 1/3 → 1) — enough fidelity levels to
  show screening without spending the whole budget climbing.
- **16 seeds** in the harness/gate (8 in the fast `pytest` direction checks) keeps the mean
  advantages stable while the run stays under ~15s.

## Optuna as an optional cross-check, never a dependency

Like MLForge's optional scikit-learn cross-check, the `[optuna]` extra re-runs `random` and
`tpe` with Optuna's own sampler and confirms the same ranking. It is never imported on the
default path, so CI stays green with numpy alone; the tests `importorskip` it.

## BLAS thread pinning

Carried over from MLForge and just as load-bearing: the benchmark is many tiny linear-
algebra calls, where multi-threaded BLAS contention is slower and non-deterministic. One
thread, pinned before numpy imports, in `__init__.py`, the CI env, and `conftest.py`.

## Known limitations / future work

- Surfaces are 2-D. Higher dimensions would widen the gap GP-EI enjoys and stress TPE; the
  machinery is dimension-general, only the bundled surfaces are 2-D.
- No real training curves: the multi-fidelity proxy is synthetic. A real-data variant (e.g.
  epochs of a small model) would be a natural extension, at the cost of offline determinism.
- The GP uses a fixed length-scale rather than marginal-likelihood tuning — deliberately,
  to keep it dependency-free and stable across seeds.
