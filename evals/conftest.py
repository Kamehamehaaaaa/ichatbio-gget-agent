"""
Pytest wiring: markers, a results log per run, and a regression diff against the saved baseline.

Results go to evals/results/run-<timestamp>.json. With --save-baseline, the run is also copied to
evals/results/baseline.json. Every run is compared to the baseline and prints which cases newly
fail (regressions) or newly pass (fixes).
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"
BASELINE = RESULTS_DIR / "baseline.json"

sys.path.insert(0, str(EVALS_DIR))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

_results: list[dict] = []


def pytest_addoption(parser):
    parser.addoption("--save-baseline", action="store_true", help="Save this run as evals/results/baseline.json")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: BLAST / NCBI Virus cases that take minutes")


@pytest.fixture(scope="session")
def results_log():
    return _results


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _results:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_file = RESULTS_DIR / f"run-{stamp}.json"
    payload = {"timestamp": stamp, "cases": _results}
    run_file.write_text(json.dumps(payload, indent=2, default=str))

    tr = terminalreporter
    tr.section("gget agent regression summary")
    passed = sum(r["passed"] for r in _results)
    tr.write_line(f"{passed}/{len(_results)} cases passed   (results: {run_file.relative_to(EVALS_DIR.parent)})")

    # Per-metric pass rates
    by_metric: dict[str, list[bool]] = {}
    for r in _results:
        for name, m in r["metrics"].items():
            by_metric.setdefault(name, []).append(m["passed"])
    for name, oks in by_metric.items():
        tr.write_line(f"  {name:<22} {sum(oks)}/{len(oks)}")

    # Regression diff vs baseline
    if BASELINE.exists():
        base = {c["id"]: c["passed"] for c in json.loads(BASELINE.read_text())["cases"]}
        now = {r["id"]: r["passed"] for r in _results}
        regressions = sorted(i for i in now if base.get(i) is True and not now[i])
        fixes = sorted(i for i in now if base.get(i) is False and now[i])
        new = sorted(i for i in now if i not in base)
        tr.write_line("")
        tr.write_line(f"vs baseline: {len(regressions)} regression(s), {len(fixes)} fix(es), {len(new)} new case(s)")
        for i in regressions:
            tr.write_line(f"  REGRESSION  {i}", red=True)
        for i in fixes:
            tr.write_line(f"  FIXED       {i}", green=True)
    else:
        tr.write_line("No baseline yet; run with --save-baseline once results look right.")

    if config.getoption("--save-baseline"):
        BASELINE.write_text(json.dumps(payload, indent=2, default=str))
        tr.write_line(f"Saved baseline: {BASELINE.relative_to(EVALS_DIR.parent)}")
