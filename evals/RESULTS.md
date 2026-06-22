# TuneForge — offline benchmark results

Seeds: **16** · budget: **60** full-fidelity-equivalent evaluations · all numbers vs the analytic optimum, no API keys, no downloads.

## Effect 1 — sample-efficiency (single fidelity)

Mean simple regret over seeds; **advantage** = random / optimizer (>1 means it beats random search).

| surface | structure | random | TPE | GP-EI | TPE adv | GP-EI adv |
|---|---|--:|--:|--:|--:|--:|
| branin | smooth | 0.8729 | 0.2096 | 0.0539 | 4.16x | 16.20x |
| ackley | rugged+basin | 2.8661 | 1.6329 | 1.3227 | 1.76x | 2.17x |
| null | **none (control)** | 0.0155 | 0.0320 | 0.0139 | 0.48x | 1.11x |

Model-based search converts structure into a large advantage on branin/ackley; on the structureless null surface the advantage collapses to ~1x — proving the edge comes from exploitable structure, not the algorithm alone.

## Effect 2 — budget-efficiency (multi fidelity)

Mean regret-vs-budget AUC (lower is better); **advantage** = random / optimizer.

| surface | fidelity proxy | random | succ.halving | Hyperband | SH adv | HB adv |
|---|---|--:|--:|--:|--:|--:|
| branin | rank-correlated | 3.247 | 1.918 | 2.460 | 1.69x | 1.32x |
| branin | **decorrelated (ablation)** | 3.247 | 5.611 | 5.094 | 0.58x | 0.64x |
| ackley | rank-correlated | 4.265 | 3.690 | 3.761 | 1.16x | 1.13x |
| ackley | **decorrelated (ablation)** | 4.265 | 4.780 | 4.425 | 0.89x | 0.96x |

With a rank-correlated cheap proxy, screening at low fidelity pays off and the multi-fidelity optimizers beat random per unit budget. Decorrelate the proxy and the early eliminations become random — the advantage collapses below 1x.

> Reproduce: `python -m evals.harness` (writes this file), `python -m evals.gate` (asserts the dissociation).
