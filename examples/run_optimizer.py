"""Minimal example: run every optimizer on Branin and print simple regret.

    python examples/run_optimizer.py
"""

from __future__ import annotations

from tuneforge.metrics import auc_regret
from tuneforge.optimizers import OPTIMIZERS
from tuneforge.runner import run
from tuneforge.types import Config

SURFACE = "branin"
BUDGET = 60.0
SEED = 0


def main() -> None:
    print(f"Branin (optimum 0.397887), budget {BUDGET:.0f}, seed {SEED}\n")
    print(f"{'optimizer':20s} {'regret':>10s} {'auc':>10s}")
    for name in OPTIMIZERS:
        r = run(Config(name, SURFACE, "correlated", BUDGET, SEED, "numpy"))
        print(f"{name:20s} {r.simple_regret:10.4f} {auc_regret(r, cap=30.0):10.4f}")


if __name__ == "__main__":
    main()
