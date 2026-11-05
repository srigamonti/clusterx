"""Keep test-generated files in pytest's per-test temporary directories."""

import pytest


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
