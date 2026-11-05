"""Test commands must keep generated artifacts out of the caller's directory."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entrypoint,with_reports", [("pytest", False), ("cell", False), ("pytest", True)])
def test_test_runs_isolate_artifacts(tmp_path, entrypoint, with_reports):
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    marker = launch_dir / "keep.txt"
    marker.write_text("Existing files must remain untouched.\n")
    before = {path.name: path.read_bytes() for path in launch_dir.iterdir()}
    artifacts_dir = tmp_path / "artifacts"
    env = os.environ.copy()
    env["PYTEST_ADDOPTS"] = shlex.join(["--basetemp", str(artifacts_dir), "-p", "no:cacheprovider"])
    if entrypoint == "pytest":
        test_path = Path(__file__).with_name("test_parent_lattice.py")
        command = [sys.executable, "-m", "pytest", "-q", str(test_path)]
    else:
        command = ["cell", "test", "-n", "test_parent_lattice"]
    reports_dir = tmp_path / "reports"
    if with_reports:
        command.extend([
            "--junitxml", str(reports_dir / "testlog.xml"),
            "--html", str(reports_dir / "testlog.html"),
        ])

    result = subprocess.run(command, cwd=launch_dir, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name: path.read_bytes() for path in launch_dir.iterdir()} == before
    for filename in ("test_parent_lattice_creation_0.json", "test_parent_lattice_creation_1.json"):
        assert len(list(artifacts_dir.rglob(filename))) == 1
    if with_reports:
        assert (reports_dir / "testlog.xml").is_file()
        assert (reports_dir / "testlog.html").is_file()
