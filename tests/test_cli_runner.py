"""CLI subcommands run, and the runner wires surface + optimizer correctly."""

from __future__ import annotations

import numpy as np

from tuneforge.cli import main as cli_main
from tuneforge.runner import run
from tuneforge.surfaces import make_surface
from tuneforge.types import Config


def test_runner_sets_optimum_from_surface():
    r = run(Config("random", "ackley", "correlated", 20.0, 0, "numpy"))
    assert r.optimum == make_surface("ackley").optimum
    assert r.surface == "ackley"


def test_runner_incumbent_is_best_full_fidelity_value():
    r = run(Config("random", "branin", "correlated", 20.0, 0, "numpy"))
    best = min(t.value for t in r.trials if t.fidelity >= 1.0)
    assert r.incumbent_value == best


def test_cli_surfaces(capsys):
    cli_main(["surfaces"])
    out = capsys.readouterr().out
    assert "branin" in out and "ackley" in out and "null" in out


def test_cli_compare(capsys):
    cli_main(["compare", "--surface", "branin", "--budget", "30", "--seed", "0"])
    out = capsys.readouterr().out
    assert "random" in out and "hyperband" in out


def test_cli_optimize(capsys):
    cli_main(["optimize", "--optimizer", "tpe", "--surface", "branin",
              "--budget", "30", "--seed", "0"])
    out = capsys.readouterr().out
    assert "regret" in out and "incumbent" in out


def test_cli_output_is_ascii(capsys):
    cli_main(["compare", "--surface", "ackley", "--budget", "30", "--seed", "1"])
    out = capsys.readouterr().out
    out.encode("ascii")  # raises if any non-ASCII leaked to the console
    assert np.isfinite(0.0)  # sanity
