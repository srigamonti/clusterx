"""Package imports and CLI startup must leave legacy configuration files alone."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("legacy_contents", [
    pytest.param(None, id="no-config"),
    pytest.param(b"not valid JSON\n", id="invalid-config"),
    pytest.param(
        b'[{"GENERAL": {"PRECISION": "3", "PBC": {"a1": true, "a2": true, "a3": false}}}]\n',
        id="custom-config",
    ),
])
@pytest.mark.parametrize("command,expected_output", [
    pytest.param(
        [sys.executable, "-c", "import clusterx"],
        "",
        id="import",
    ),
    pytest.param(["cell", "--help"], "CELL Command Line Interface", id="help"),
    pytest.param(["cell", "test", "-l"], "test_parent_lattice", id="direct-command"),
    pytest.param(["cell", "--input_file", "workflow.toml"], "test_parent_lattice", id="toml-command"),
])
def test_startup_does_not_use_cellconfig(tmp_path, legacy_contents, command, expected_output):
    if legacy_contents is not None:
        (tmp_path / "cellconfig.json").write_bytes(legacy_contents)
    if "--input_file" in command:
        (tmp_path / "workflow.toml").write_text("[test]\ntestslist = true\n")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    if expected_output:
        assert expected_output in result.stdout
    else:
        assert result.stdout == ""
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
