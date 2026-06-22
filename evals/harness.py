"""Run the full offline benchmark and write ``evals/RESULTS.md``.

Two experiments, both scored against the analytic optimum:

1. **Sample-efficiency** (single fidelity) -- random vs TPE vs GP-EI on a
   structured surface (branin), a rugged-but-structured one (ackley), and the
   structureless control (null). Model-based search should win on the first two and
   collapse to random on the third.
2. **Budget-efficiency** (multi fidelity) -- random vs successive-halving vs
   Hyperband, under a rank-``correlated`` cheap proxy and a ``decorrelated`` one.
   The multi-fidelity edge should survive the first and collapse under the second.
"""

from __future__ import annotations

import io
from pathlib import Path

from evals.metrics import advantage, mean_auc, mean_regret

SEEDS = 16
BUDGET = 60.0
CAPS = {"branin": 30.0, "ackley": 22.0, "null": 1.0}

SAMPLE_SURFACES = ["branin", "ackley", "null"]
MODEL_BASED = ["tpe", "gp_ei"]
MULTI_SURFACES = ["branin", "ackley"]
MULTI_OPT = ["successive_halving", "hyperband"]
REGIMES = ["correlated", "decorrelated"]

RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


def compute() -> dict:
    sample: dict[str, dict] = {}
    for surf in SAMPLE_SURFACES:
        base = mean_regret("random", surf, BUDGET, SEEDS)
        row = {"random": {"regret": base, "advantage": 1.0}}
        for opt in MODEL_BASED:
            r = mean_regret(opt, surf, BUDGET, SEEDS)
            row[opt] = {"regret": r, "advantage": advantage(base, r)}
        sample[surf] = row

    multi: dict[str, dict] = {}
    for surf in MULTI_SURFACES:
        cap = CAPS[surf]
        multi[surf] = {}
        for regime in REGIMES:
            base = mean_auc("random", surf, regime, BUDGET, SEEDS, cap)
            row = {"random": {"auc": base, "advantage": 1.0}}
            for opt in MULTI_OPT:
                a = mean_auc(opt, surf, regime, BUDGET, SEEDS, cap)
                row[opt] = {"auc": a, "advantage": advantage(base, a)}
            multi[surf][regime] = row

    return {"sample": sample, "multi": multi,
            "meta": {"seeds": SEEDS, "budget": BUDGET, "caps": CAPS}}


def _render(results: dict) -> str:
    out = io.StringIO()
    w = out.write
    m = results["meta"]
    w("# TuneForge — offline benchmark results\n\n")
    w(f"Seeds: **{m['seeds']}** · budget: **{m['budget']:.0f}** full-fidelity-equivalent "
      "evaluations · all numbers vs the analytic optimum, no API keys, no downloads.\n\n")

    w("## Effect 1 — sample-efficiency (single fidelity)\n\n")
    w("Mean simple regret over seeds; **advantage** = random / optimizer "
      "(>1 means it beats random search).\n\n")
    w("| surface | structure | random | TPE | GP-EI | TPE adv | GP-EI adv |\n")
    w("|---|---|--:|--:|--:|--:|--:|\n")
    labels = {"branin": "smooth", "ackley": "rugged+basin", "null": "**none (control)**"}
    for surf in SAMPLE_SURFACES:
        r = results["sample"][surf]
        w(f"| {surf} | {labels[surf]} | {r['random']['regret']:.4f} | "
          f"{r['tpe']['regret']:.4f} | {r['gp_ei']['regret']:.4f} | "
          f"{r['tpe']['advantage']:.2f}x | {r['gp_ei']['advantage']:.2f}x |\n")
    w("\nModel-based search converts structure into a large advantage on branin/ackley; "
      "on the structureless null surface the advantage collapses to ~1x — proving the "
      "edge comes from exploitable structure, not the algorithm alone.\n\n")

    w("## Effect 2 — budget-efficiency (multi fidelity)\n\n")
    w("Mean regret-vs-budget AUC (lower is better); **advantage** = random / optimizer.\n\n")
    w("| surface | fidelity proxy | random | succ.halving | Hyperband | SH adv | HB adv |\n")
    w("|---|---|--:|--:|--:|--:|--:|\n")
    for surf in MULTI_SURFACES:
        for regime in REGIMES:
            r = results["multi"][surf][regime]
            tag = "rank-correlated" if regime == "correlated" else "**decorrelated (ablation)**"
            w(f"| {surf} | {tag} | {r['random']['auc']:.3f} | "
              f"{r['successive_halving']['auc']:.3f} | {r['hyperband']['auc']:.3f} | "
              f"{r['successive_halving']['advantage']:.2f}x | "
              f"{r['hyperband']['advantage']:.2f}x |\n")
    w("\nWith a rank-correlated cheap proxy, screening at low fidelity pays off and the "
      "multi-fidelity optimizers beat random per unit budget. Decorrelate the proxy and the "
      "early eliminations become random — the advantage collapses below 1x.\n\n")

    w("> Reproduce: `python -m evals.harness` (writes this file), "
      "`python -m evals.gate` (asserts the dissociation).\n")
    return out.getvalue()


def main() -> None:
    results = compute()
    RESULTS_PATH.write_text(_render(results), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    # ASCII-only console summary (Windows cp1252 safe).
    for surf in SAMPLE_SURFACES:
        r = results["sample"][surf]
        print(f"sample  {surf:8s} random={r['random']['regret']:.4f} "
              f"tpe_adv={r['tpe']['advantage']:.2f}x gp_adv={r['gp_ei']['advantage']:.2f}x")
    for surf in MULTI_SURFACES:
        for regime in REGIMES:
            r = results["multi"][surf][regime]
            print(f"budget  {surf:7s} {regime:12s} "
                  f"sh_adv={r['successive_halving']['advantage']:.2f}x "
                  f"hb_adv={r['hyperband']['advantage']:.2f}x")


if __name__ == "__main__":
    main()
