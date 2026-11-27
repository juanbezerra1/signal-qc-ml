"""What gets installed: the version, the typing marker, the two queries, the command, and nothing that is not code."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

import signal_qc_ml
from signal_qc_ml import __version__

from .helpers import ROOT

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PACKAGE = Path(signal_qc_ml.__file__).parent


def test_the_version_is_the_one_in_pyproject() -> None:
    assert __version__ == PYPROJECT["project"]["version"]


def test_the_package_says_that_it_has_types() -> None:
    assert (PACKAGE / "py.typed").is_file()


def test_the_two_queries_ship_with_the_package() -> None:
    assert {p.name for p in PACKAGE.glob("*.sql")} == {"recordings.sql", "signals.sql"}


def test_the_command_runs_as_a_module(tmp_path: Path) -> None:
    # from another folder, and with the package where this test imported it from (installed or not)
    env = {**os.environ, "PYTHONPATH": str(PACKAGE.parent)}
    done = subprocess.run(
        [sys.executable, "-m", "signal_qc_ml", "--version"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert done.stdout.strip() == f"signal-qc-ml {__version__}"


def test_the_console_script_points_at_a_function_that_exists() -> None:
    target = PYPROJECT["project"]["scripts"]["signal-qc-ml"]
    module, _, function = target.partition(":")
    assert module == "signal_qc_ml.cli"
    assert callable(getattr(importlib.import_module(module), function))


@pytest.mark.skipif(importlib.util.find_spec("hatchling") is None, reason="hatchling is not installed")
def test_the_wheel_has_the_code_and_the_queries_and_nothing_else(tmp_path: Path) -> None:
    built = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = next(archive.read(n).decode("utf-8") for n in names if n.endswith(".dist-info/METADATA"))
        entry_points = next(archive.read(n).decode("utf-8") for n in names if n.endswith("entry_points.txt"))
    inside = {n for n in names if n.startswith("signal_qc_ml/")}
    expected = {f"signal_qc_ml/{p.name}" for p in PACKAGE.iterdir() if p.suffix in {".py", ".sql", ".typed"}}
    assert inside == expected
    assert [n for n in names if not n.startswith("signal_qc_ml/") and ".dist-info/" not in n] == []
    assert f"Version: {__version__}" in metadata
    assert "Requires-Python: >=3.11" in metadata
    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:") and "extra ==" not in line]
    assert len(requires) == len(PYPROJECT["project"]["dependencies"])
    assert "signal-qc-ml = signal_qc_ml.cli:main" in entry_points
