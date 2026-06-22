"""Run one optimizer on one surface and assemble an :class:`OptResult`.

The recommended incumbent is the best config ever evaluated at full fidelity
(``b == 1``), whose value is exact. Multi-fidelity optimizers always finish a
bracket at full fidelity, so they always have a valid incumbent; an optimizer that
spent everything on cheap looks would be correctly left with infinite regret.
"""

from __future__ import annotations

from tuneforge.config import Settings
from tuneforge.optimizers import make_optimizer
from tuneforge.problem import Problem
from tuneforge.surfaces import make_surface
from tuneforge.types import Config, OptResult


def run(config: Config) -> OptResult:
    surface = make_surface(config.surface)
    settings = Settings(
        optimizer=config.optimizer,
        surface=config.surface,
        fidelity_regime=config.fidelity_regime,
        budget=config.budget,
        seed=config.seed,
        backend=config.backend,
    )

    if config.backend == "optuna":
        from tuneforge.optuna_check import run_optuna

        return run_optuna(config, surface)

    problem = Problem(surface, budget=config.budget, regime=config.fidelity_regime)
    rng = settings.rng(0xABC, _optimizer_offset(config.optimizer))
    optimize = make_optimizer(config.optimizer)
    optimize(problem, rng)

    result = OptResult(
        optimizer=config.optimizer,
        surface=config.surface,
        seed=config.seed,
        budget=config.budget,
        trials=problem.trials,
        optimum=surface.optimum,
    )
    _set_incumbent(result)
    return result


def _set_incumbent(result: OptResult) -> None:
    best_x, best_v = None, float("inf")
    for t in result.trials:
        if t.fidelity >= 1.0 and t.value < best_v:
            best_v, best_x = t.value, t.x
    result.incumbent_x = best_x
    result.incumbent_value = best_v


def _optimizer_offset(name: str) -> int:
    # Distinct per-optimizer RNG stream so they don't share the same random draws.
    return sum(ord(c) for c in name)


def run_settings(settings: Settings) -> OptResult:
    return run(settings.to_config())
