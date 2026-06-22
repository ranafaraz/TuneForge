"""The quality gate passes on the shipped code (full-strength, all thresholds)."""

from __future__ import annotations

from evals.gate import run_checks


def test_gate_all_checks_pass():
    checks = run_checks()
    failed = [msg for ok, msg in checks if not ok]
    assert not failed, "gate regressions: " + "; ".join(failed)
    assert len(checks) >= 15
