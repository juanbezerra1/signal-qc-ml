"""Shared by every test: the project folder is the same after the tests as it was before them."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IGNORED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}


def files_of(root: Path) -> set[str]:
    """The files under ``root``, as paths relative to it, leaving out the folders that tools fill by themselves."""
    found: set[str] = set()
    for folder, names, files in os.walk(root):
        names[:] = [n for n in names if n not in IGNORED and not n.endswith(".egg-info")]
        found.update(os.path.relpath(os.path.join(folder, name), root) for name in files)
    return found


@pytest.fixture(scope="session", autouse=True)
def nothing_is_left_in_the_project() -> Iterator[None]:
    """A test that writes where it runs leaves files in the project (and in the next zip): fail when one does."""
    before = files_of(ROOT)
    yield
    left = sorted(files_of(ROOT) - before)
    assert not left, f"the tests left files in the project folder: {', '.join(left)}"
