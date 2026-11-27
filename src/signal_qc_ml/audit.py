"""Every split of the evaluation, looked at by ``table-audit`` before its numbers are believed.

The evaluation cuts the table by participant on purpose; this is the second pair of eyes. For each split the train
and the test side are written as two CSV files and given to ``table-audit leakage``, which looks for a participant on
both sides, for a row that is in both, and for a column that gives the target away. Nothing here decides what the
audit looks for: that is the job of ``table-audit``, and it is a separate project.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import columns as cols
from .cv import LeakageError, Split

DROPPED = ("recording_id", "session_id", "participant_id")  # ids: a row with an id of its own is never a copy


class AuditUnavailable(RuntimeError):
    """``table-audit`` is not installed (or does not run), so the splits cannot be audited."""


@dataclass(frozen=True)
class SplitAudit:
    kind: str  # "grouped" or "random"
    repeat: int
    fold: int
    errors: dict[str, int]  # check -> how many rows or participants it concerns
    warnings: dict[str, int]
    version: str
    messages: tuple[str, ...] = ()  # what the audit said about each error, in its words

    @property
    def failed(self) -> bool:
        return bool(self.errors)


@dataclass
class AuditSummary:
    """What the audit said about a set of splits."""

    kind: str
    version: str
    splits: int = 0
    failed: int = 0
    checks: Counter[str] = field(default_factory=Counter)  # check -> in how many splits it appeared
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": f"table-audit {self.version}",
            "kind": self.kind,
            "splits": self.splits,
            "splits_with_errors": self.failed,
            "checks": dict(sorted(self.checks.items())),
            "first_messages": self.messages[:3],
        }


def check_available() -> None:
    """Raise ``AuditUnavailable`` unless ``table_audit`` can be imported by this Python."""
    if importlib.util.find_spec("table_audit") is None:
        raise AuditUnavailable(
            "table-audit is not installed, and the splits are audited with it before the results are believed.\n"
            "Install it (see the README) or run again with --skip-audit to go without; the report will say so."
        )


def _write(frame: pd.DataFrame, rows: Any, path: Path) -> None:
    keep = [c for c in frame.columns if c not in DROPPED]
    frame.iloc[rows][keep].to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def audit_split(frame: pd.DataFrame, split: Split, kind: str, workdir: Path) -> SplitAudit:
    """Write the two sides of ``split`` and run ``table-audit leakage`` on them."""
    train, test = workdir / "train.csv", workdir / "test.csv"
    _write(frame, split.train, train)
    _write(frame, split.test, test)
    command = [
        sys.executable,
        "-m",
        "table_audit",
        "leakage",
        str(train),
        str(test),
        "--group",
        cols.GROUP_LABEL,
        "--target",
        cols.TARGET,
        "--format",
        "json",
    ]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    if done.returncode not in (0, 1):  # 1 means "found errors"; anything else means it could not look
        raise AuditUnavailable(f"table-audit could not run (exit code {done.returncode}): {done.stderr.strip()[:300]}")
    report = json.loads(done.stdout)
    errors: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for finding in report["findings"]:
        bucket = errors if finding["severity"] == "error" else warnings if finding["severity"] == "warning" else None
        if bucket is not None:
            bucket[finding["check"]] = bucket.get(finding["check"], 0) + int(finding["count"])
    said = tuple(f["message"] for f in report["findings"] if f["severity"] == "error")
    return SplitAudit(kind, split.repeat, split.fold, errors, warnings, str(report["version"]), said)


def audit_splits(frame: pd.DataFrame, splits: list[Split], kind: str) -> AuditSummary:
    """Audit every split in ``splits``; the summary says in how many of them the audit found errors, and which."""
    check_available()
    summary = AuditSummary(kind, "")
    with tempfile.TemporaryDirectory(prefix="signal-qc-ml-audit-") as name:
        for split in splits:
            result = audit_split(frame, split, kind, Path(name))
            summary.version = result.version
            summary.splits += 1
            summary.failed += int(result.failed)
            for check in (*result.errors, *result.warnings):
                summary.checks[check] += 1
            summary.messages.extend(result.messages)
    return summary


def require_clean(summary: AuditSummary) -> None:
    """Raise ``LeakageError`` if the audit found an error in any split that the evaluation relies on."""
    if summary.failed:
        raise LeakageError(
            f"table-audit found errors in {summary.failed} of {summary.splits} splits: "
            f"{', '.join(sorted(summary.checks))}. {summary.messages[0] if summary.messages else ''}".strip()
        )
