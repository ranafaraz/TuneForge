# TuneForge — agent guide

A from-scratch benchmark of black-box hyperparameter optimizers against **known optima**.
The product is a *measurement*, not a tuning library: it proves a two-effect dissociation,
each effect demonstrated by an ablation that collapses it.

> This file is the canonical agent guide. `AGENTS.md` mirrors it for non-Claude tools —
> **edit both together**.

## The thesis (don't dilute it)
- **Model-based search (TPE, GP-EI) buys sample-efficiency from response-surface
  structure.** Ablation: on the structureless `null` surface the advantage over random
  collapses (branin GP-EI 16.2× → null ~1.1×).
- **Multi-fidelity search (successive halving, Hyperband) buys budget-efficiency from
  fidelity-rank correlation.** Ablation: with the `decorrelated` proxy the advantage
  collapses below random (branin SH 1.69× → 0.58×).
- The two ablations are **orthogonal** — structure removal leaves multi-fidelity untouched,
  decorrelation leaves sample-efficiency untouched. Keep them that way.

## Invariants (do not regress)
1. **One black box, one budget axis.** Optimizers only ever call `Problem.evaluate(x, b)`;
   cost == fidelity. Regret is read **only** off `b == 1` (exact) evaluations, so single-
   and multi-fidelity methods compete on total budget spent. Don't let an optimizer score
   off low-fidelity values.
2. **Fidelity model is exact at `b = 1`** and identical across regimes (`surfaces/base.py`).
   The regime may change the search *path*, never the final score. This is what makes the
   multi-fidelity ablation clean — preserve it.
3. **`null` must stay structureless** (independent per-cell values). It's the control that
   proves model-based wins come from structure; if a model ever beats random on `null`,
   that's a bug, not an improvement.
4. **Determinism.** All RNG via `Settings.rng(SALT, seed, *offsets)`, never `hash()`/clock.
   `default_rng` is cross-version stable, so eval numbers are identical on CI — the gate
   relies on this for tight thresholds.
5. **Pin BLAS to one thread** before numpy imports (`tuneforge/__init__.py`, CI `env:`,
   `tests/conftest.py`). Many tiny matrix ops; multi-threaded BLAS is slower and
   non-deterministic. Never remove.
6. **Optimizers honest, not strawmen.** TPE uses adaptive per-point bandwidths (a single
   global bandwidth makes it *worse* than random — that was a real bug). Don't "tune" an
   optimizer to clear the gate; tune the experiment design if a margin is too thin.
7. **Gate asserts shape, not luck** (`evals/gate.py`): wins where expected, collapses where
   expected. 19 checks. Keep it the source of truth for "did I break the thesis."
8. **Offline-first.** numpy is the only runtime dep. Optuna is an optional `[optuna]`
   cross-check, `importorskip`-ed in tests, never on the default path. Windows console is
   cp1252 — CLI prints ASCII only; the harness writes UTF-8 to `evals/RESULTS.md`.

## Layout
`tuneforge/` core (types, config, spaces, problem, surfaces/, optimizers/, runner, metrics,
cli, optuna_check) · `evals/` (metrics aggregate, harness→RESULTS.md, gate) · `tests/`
(56 pass + optuna skipped without the extra) · `examples/run_optimizer.py` · `docs/`
(ARCHITECTURE, DECISIONS, demo.gif 1×1 placeholder).

## Run offline
```bash
pip install -e ".[dev]"
pytest -q                 # 56 tests (+ optuna skipped without [optuna])
ruff check .
python -m evals.harness   # writes evals/RESULTS.md
python -m evals.gate      # asserts the dissociation (19/19)
tuneforge compare --surface branin --budget 60 --seed 0
```

## Commit policy (hard rule)
Author = **Rana Faraz only**. Never add a `Co-Authored-By: Claude` trailer or any AI
branding. Keep history tidy and incremental (one logical change per commit).
