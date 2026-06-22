"""Quality gate: fail CI unless the two-effect dissociation actually holds.

The gate asserts the *shape* of the result, not just a single lucky number:

* model-based search wins big on structured surfaces and *not at all* on the
  structureless control (sample-efficiency, and its ablation);
* multi-fidelity search beats random under a rank-correlated proxy and *loses* to it
  under a decorrelated proxy (budget-efficiency, and its ablation).

Thresholds sit well inside the measured margins; `np.random.default_rng` is stable
across platforms and Python versions, so the numbers are identical on CI.
"""

from __future__ import annotations

import sys

from evals.harness import compute

# --- sample-efficiency thresholds ---
BRANIN_TPE_MIN = 2.5      # model-based clearly beats random on a structured surface
BRANIN_GP_MIN = 6.0
ACKLEY_MODEL_MIN = 1.4    # still wins on the rugged-but-structured surface
NULL_MODEL_MAX = 1.5      # advantage collapses on the structureless control
DISSOC_FACTOR = 2.0       # structured advantage >= this x null advantage

# --- budget-efficiency thresholds ---
CORR_MULTI_MIN = 1.05     # beats random per unit budget under a correlated proxy
BRANIN_CORR_SH_MIN = 1.3  # strong headline win
DECORR_BRANIN_MAX = 0.9   # collapses below random when the proxy is decorrelated
COLLAPSE_MARGIN = 0.10    # correlated advantage exceeds decorrelated by this much


def _check(checks: list[tuple[bool, str]], ok: bool, msg: str) -> None:
    checks.append((ok, msg))


def run_checks() -> list[tuple[bool, str]]:
    r = compute()
    s = r["sample"]
    m = r["multi"]
    checks: list[tuple[bool, str]] = []

    # ---- Effect 1: sample-efficiency + ablation ----
    b_tpe = s["branin"]["tpe"]["advantage"]
    b_gp = s["branin"]["gp_ei"]["advantage"]
    _check(checks, b_tpe >= BRANIN_TPE_MIN,
           f"branin TPE advantage {b_tpe:.2f}x >= {BRANIN_TPE_MIN}")
    _check(checks, b_gp >= BRANIN_GP_MIN,
           f"branin GP-EI advantage {b_gp:.2f}x >= {BRANIN_GP_MIN}")
    for opt in ("tpe", "gp_ei"):
        a = s["ackley"][opt]["advantage"]
        _check(checks, a >= ACKLEY_MODEL_MIN,
               f"ackley {opt} advantage {a:.2f}x >= {ACKLEY_MODEL_MIN}")
        nadv = s["null"][opt]["advantage"]
        _check(checks, nadv <= NULL_MODEL_MAX,
               f"null {opt} advantage {nadv:.2f}x <= {NULL_MODEL_MAX} (collapse)")
        badv = s["branin"][opt]["advantage"]
        _check(checks, badv >= DISSOC_FACTOR * nadv,
               f"{opt}: branin adv {badv:.2f}x >= {DISSOC_FACTOR}x null adv {nadv:.2f}x")

    # ---- Effect 2: budget-efficiency + ablation ----
    for surf in ("branin", "ackley"):
        for opt in ("successive_halving", "hyperband"):
            corr = m[surf]["correlated"][opt]["advantage"]
            decorr = m[surf]["decorrelated"][opt]["advantage"]
            _check(checks, corr >= CORR_MULTI_MIN,
                   f"{surf} {opt} correlated advantage {corr:.2f}x >= {CORR_MULTI_MIN}")
            _check(checks, corr >= decorr + COLLAPSE_MARGIN,
                   f"{surf} {opt}: correlated {corr:.2f}x > decorrelated {decorr:.2f}x (collapse)")
    sh_b = m["branin"]["correlated"]["successive_halving"]["advantage"]
    _check(checks, sh_b >= BRANIN_CORR_SH_MIN,
           f"branin SH correlated advantage {sh_b:.2f}x >= {BRANIN_CORR_SH_MIN}")
    for opt in ("successive_halving", "hyperband"):
        d = m["branin"]["decorrelated"][opt]["advantage"]
        _check(checks, d <= DECORR_BRANIN_MAX,
               f"branin {opt} decorrelated advantage {d:.2f}x <= {DECORR_BRANIN_MAX} (collapse)")

    return checks


def main() -> int:
    checks = run_checks()
    passed = 0
    for ok, msg in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
        passed += ok
    total = len(checks)
    print(f"\nGate: {passed}/{total} checks passed.")
    if passed == total:
        print("PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
