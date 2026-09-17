"""Package version: single source mirrored from pyproject (no importlib cost)."""

import tomllib
from pathlib import Path

import torchmamba


def test_version_matches_pyproject():
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    )
    assert torchmamba.__version__ == pyproject["project"]["version"]
