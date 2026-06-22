"""Command-line interface. Prints ASCII only (Windows consoles are cp1252)."""

from __future__ import annotations

import argparse

from tuneforge.config import Settings
from tuneforge.metrics import auc_regret
from tuneforge.optimizers import MODEL_BASED, MULTI_FIDELITY, OPTIMIZERS
from tuneforge.runner import run
from tuneforge.surfaces import SURFACES, make_surface
from tuneforge.types import Config


def _build_config(args: argparse.Namespace) -> Config:
    s = Settings.from_env()
    return Config(
        optimizer=getattr(args, "optimizer", None) or s.optimizer,
        surface=getattr(args, "surface", None) or s.surface,
        fidelity_regime=getattr(args, "fidelity", None) or s.fidelity_regime,
        budget=getattr(args, "budget", None) or s.budget,
        seed=getattr(args, "seed", None) if getattr(args, "seed", None) is not None else s.seed,
        backend=getattr(args, "backend", None) or s.backend,
    )


def _cmd_optimize(args: argparse.Namespace) -> None:
    cfg = _build_config(args)
    result = run(cfg)
    print(f"optimizer : {cfg.optimizer}")
    print(f"surface   : {cfg.surface}  (optimum = {result.optimum:.6f})")
    print(f"fidelity  : {cfg.fidelity_regime}")
    print(f"budget    : {cfg.budget:.1f}   spent {result.spent:.1f}   trials {result.n_trials}")
    print(f"incumbent : value {result.incumbent_value:.6f}")
    print(f"regret    : {result.simple_regret:.6f}")


def _cmd_compare(args: argparse.Namespace) -> None:
    surf = make_surface(args.surface)
    cap = float(args.cap)
    print(f"surface {args.surface}  optimum={surf.optimum:.6f}  "
          f"budget={args.budget:.0f}  seed={args.seed}  regime={args.fidelity}")
    print(f"{'optimizer':20s} {'family':14s} {'regret':>10s} {'auc':>10s} {'spent':>8s}")
    print("-" * 66)
    for name in OPTIMIZERS:
        cfg = Config(name, args.surface, args.fidelity, args.budget, args.seed, "numpy")
        r = run(cfg)
        fam = "multi-fidelity" if name in MULTI_FIDELITY else (
            "model-based" if name in MODEL_BASED else "baseline")
        print(f"{name:20s} {fam:14s} {r.simple_regret:10.4f} "
              f"{auc_regret(r, cap=cap):10.4f} {r.spent:8.1f}")


def _cmd_surfaces(_args: argparse.Namespace) -> None:
    print(f"{'surface':12s} {'dim':>4s} {'optimum':>12s}")
    print("-" * 30)
    for name in SURFACES:
        s = make_surface(name)
        print(f"{name:12s} {s.dim:4d} {s.optimum:12.6f}")


def _cmd_eval(_args: argparse.Namespace) -> None:
    from evals.harness import main as harness_main

    harness_main()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tuneforge", description="Black-box HPO benchmark.")
    sub = p.add_subparsers(dest="command", required=True)

    po = sub.add_parser("optimize", help="run one optimizer on one surface")
    po.add_argument("--optimizer", choices=list(OPTIMIZERS))
    po.add_argument("--surface", choices=list(SURFACES))
    po.add_argument("--fidelity", choices=["correlated", "decorrelated"])
    po.add_argument("--budget", type=float)
    po.add_argument("--seed", type=int)
    po.add_argument("--backend", choices=["numpy", "optuna"])
    po.set_defaults(func=_cmd_optimize)

    pc = sub.add_parser("compare", help="compare all optimizers on a surface")
    pc.add_argument("--surface", default="branin", choices=list(SURFACES))
    pc.add_argument("--fidelity", default="correlated", choices=["correlated", "decorrelated"])
    pc.add_argument("--budget", type=float, default=60.0)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--cap", type=float, default=30.0, help="regret cap for AUC")
    pc.set_defaults(func=_cmd_compare)

    ps = sub.add_parser("surfaces", help="list response surfaces and their optima")
    ps.set_defaults(func=_cmd_surfaces)

    pe = sub.add_parser("eval", help="run the full offline benchmark + write RESULTS.md")
    pe.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
