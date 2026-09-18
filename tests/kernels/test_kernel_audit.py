"""CI gate for the kernel audit (scripts/kernels/kernel_audit.py).

Runs the audit as a subprocess in float64 (the precision the audit is
written for; the session's own precision is left untouched), once per
session, and checks the verdict rows:

* the audit exits 0;
* no spec'd kernel has a hard FAIL row (documented KNOWN deviations are
  allowed);
* no KNOWN rows at all -- every shipped kernel currently passes every
  check (the former B7/B8 kernelScale deviation was fixed 2026-09-18 by
  adopting the shape-derived scales);
* a deliberately broken spec (cubic with a 1 % C perturbation) makes the
  audit fail -- the sanity check of the sanity check.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "scripts" / "kernels" / "kernel_audit.py"
SPECS = REPO_ROOT / "scripts" / "kernels" / "kernel_specs.yaml"


@pytest.fixture(scope="session")
def audit_run(tmp_path_factory):
    out_json = tmp_path_factory.mktemp("kernel_audit") / "rows.json"
    env = dict(os.environ, warpSPHCore_PRECISION="float64")
    proc = subprocess.run(
        [sys.executable, str(AUDIT), "--quiet", "--json", str(out_json)],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    rows = json.loads(out_json.read_text()) if out_json.exists() else []
    return proc.returncode, rows, proc.stdout + proc.stderr


def _spec_names():
    with open(SPECS) as f:
        return sorted(yaml.safe_load(f)["kernels"])


def test_audit_exits_zero(audit_run):
    rc, rows, output = audit_run
    assert rc == 0, f"audit exited {rc}:\n{output}"


def test_audit_covers_all_spec_kernels(audit_run):
    _, rows, _ = audit_run
    audited = {r["kernel"] for r in rows}
    for name in _spec_names():
        assert name in audited, f"spec '{name}' was not audited"


@pytest.mark.parametrize("kernel", _spec_names())
def test_no_hard_failures(kernel, audit_run):
    _, rows, _ = audit_run
    fails = [r for r in rows if r["kernel"] == kernel and r["status"] == "FAIL"]
    assert not fails, f"hard failures for {kernel}:\n" + "\n".join(
        f"  {r['check']} dim={r['dim']}: {r['detail']}" for r in fails
    )


def test_no_documented_deviations_remain(audit_run):
    """Every spec'd kernel passes every check -- there must be no KNOWN
    rows at all. (The known_issues mechanism is still exercised: a
    deviation that is NOT in the spec's known_issues hard-fails, and a
    spec entry that no longer matches a real failure raises a warning.)"""
    _, rows, _ = audit_run
    known = [r for r in rows if r["status"] == "KNOWN"]
    assert not known, f"unexpected KNOWN rows:\n" + "\n".join(
        f"  {r['kernel']} {r['check']} dim={r['dim']}: {r['detail']}" for r in known
    )


def test_broken_spec_fails_the_audit(tmp_path):
    """Canary: a 1 % C perturbation in a copy of the specs must produce a
    hard failure and a nonzero exit code."""
    with open(SPECS) as f:
        specs = yaml.safe_load(f)
    specs["kernels"]["cubic"]["C"]["3"] = "16.16/pi"  # 16/pi * 1.01
    broken = tmp_path / "broken_specs.yaml"
    broken.write_text(yaml.safe_dump(specs))
    out_json = tmp_path / "rows.json"
    env = dict(os.environ, warpSPHCore_PRECISION="float64")
    proc = subprocess.run(
        [sys.executable, str(AUDIT), "--specs", str(broken), "--only", "cubic",
         "--quiet", "--json", str(out_json)],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 1, f"audit should fail on the broken spec:\n{proc.stdout}\n{proc.stderr}"
    rows = json.loads(out_json.read_text())
    fails = [r for r in rows if r["status"] == "FAIL" and r["check"] == "normalisation"]
    assert fails, f"expected a normalisation FAIL, got:\n{proc.stdout}"
