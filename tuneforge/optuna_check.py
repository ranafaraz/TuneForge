"""Optional cross-check: re-run a single-fidelity optimizer with Optuna.

Mirrors TuneForge's own ``random`` and ``tpe`` with Optuna's ``RandomSampler`` and
``TPESampler`` on the identical surface, so the headline ranking (model-based beats
random on a structured surface) can be reproduced by an independent, battle-tested
implementation. Requires the ``[optuna]`` extra; it is never imported on the offline
default path.
"""

from __future__ import annotations

import numpy as np

from tuneforge.surfaces.base import Surface
from tuneforge.types import Config, OptResult, Trial

# Map TuneForge optimizers onto Optuna samplers (single-fidelity only).
_SAMPLERS = {"random", "tpe"}


def run_optuna(config: Config, surface: Surface) -> OptResult:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised only with the extra
        raise RuntimeError(
            "the optuna backend needs the optional dependency: pip install 'tuneforge[optuna]'"
        ) from exc

    if config.optimizer not in _SAMPLERS:
        raise ValueError(
            f"optuna cross-check supports {sorted(_SAMPLERS)}, not {config.optimizer!r}"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if config.optimizer == "tpe":
        sampler = optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=8)
    else:
        sampler = optuna.samplers.RandomSampler(seed=config.seed)

    dim = surface.space.dim
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
        return surface.true_value(x)

    study.optimize(objective, n_trials=int(config.budget), show_progress_bar=False)

    trials = [
        Trial(
            x=tuple(t.params[f"x{i}"] for i in range(dim)),
            fidelity=1.0,
            value=float(t.value),
            cost=1.0,
        )
        for t in study.trials
        if t.value is not None
    ]
    result = OptResult(
        optimizer=config.optimizer,
        surface=config.surface,
        seed=config.seed,
        budget=config.budget,
        trials=trials,
        optimum=surface.optimum,
    )
    best = min(trials, key=lambda t: t.value)
    result.incumbent_x, result.incumbent_value = best.x, best.value
    return result
