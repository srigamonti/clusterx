"""Test the CLI commands of the clusterx package.

Also demonstrate the use of pytests tmp_path and how to test CLI in general.
"""

import os
import subprocess

import pytest


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result


def err_str(command, response):
    return f"Command '{command}' failed with exit code {response.returncode}\nStdout: {response.stdout}\nStderr: {response.stderr}"


@pytest.mark.parametrize("command", ["cell -h", "cell --help"])
def test_help(command):
    response = run_command("cell -h")
    assert response.returncode == 0, err_str(command, response)


def run_build_parent_lattice(path):
    command = (
        f"cell build_parent_lattice --species=Si,Ge --filepath={path / 'plat.json'} "
    )
    response = run_command(command)
    assert response.returncode == 0, err_str(command, response)


def run_build_cpool(path):
    command = (
        "cell build_cpool 2,3 1.0,2.0,3.0 "
        f"-plf {path / 'plat.json'} "
        "--psc=2 "
        "--method=1 "
        f"-cpool_filepath={path / 'cpool.json'} "
        "--vacancy_atomic_number 0 "
    )
    response = run_command(command)
    assert response.returncode == 0, err_str(command, response)


def test_build(tmp_path):
    # this is a workaround, ideally the tests should be independent, but this
    # would only be possible with extra pytest-dependency package
    run_build_parent_lattice(tmp_path)

    # Check if the files were created
    assert os.path.exists(tmp_path / "plat.json")
