from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from signal_qc_ml import audit, cv

from .helpers import reviewed_of

if os.environ.get("CI"):
    import table_audit  # noqa: F401  (in CI the audit must be there: a skipped test would hide that it is not)
else:
    pytest.importorskip("table_audit", reason="the audit is the table-audit project; see the README to install it")


def test_splits_by_participant_have_nothing_to_report() -> None:
    data = reviewed_of(participants=40, seed=1)
    splits = cv.grouped_splits(data.y, data.groups, 4, 2, seed=0)
    summary = audit.audit_splits(data.frame, splits, "grouped")
    assert (summary.splits, summary.failed) == (8, 0)  # warnings may be there (a code that is rare); errors are not
    assert "group-overlap" not in summary.checks and "identical-rows" not in summary.checks
    audit.require_clean(summary)
    assert summary.as_dict()["tool"].startswith("table-audit ")


def test_splits_that_ignore_participants_are_an_error_in_every_split() -> None:
    data = reviewed_of(participants=40, seed=1)
    splits = cv.random_splits(data.y, 4, 1, seed=0)
    summary = audit.audit_splits(data.frame, splits, "random")
    assert (summary.splits, summary.failed) == (4, 4)
    assert (
        summary.checks["group-overlap"] == 4
    )  # a warning about a rare code may come along; this one may not be missing
    assert "participant_code" in summary.messages[0]
    with pytest.raises(cv.LeakageError, match="errors in 4 of 4 splits: group-overlap"):
        audit.require_clean(summary)


def test_a_copy_of_a_recording_on_both_sides_is_found(tmp_path: Path) -> None:
    data = reviewed_of(participants=40, seed=1)
    frame = data.frame.copy()
    frame.iloc[0] = frame.iloc[-1]  # the same recording twice, with other ids
    frame.loc[frame.index[0], ["recording_id", "session_id"]] = [10**6, 10**6]
    splits = cv.grouped_splits(data.y, data.groups, 4, 1, seed=0)
    split = next(s for s in splits if 0 in s.test)
    # the copy and its original are in different folds only when the participants differ; force the case
    both = cv.Split(0, 0, split.train, split.test)
    with_copy = frame.copy()
    with_copy.loc[with_copy.index[split.test[0]]] = with_copy.loc[with_copy.index[split.train[0]]]
    with_copy["recording_id"] = range(len(with_copy))
    with_copy["session_id"] = range(len(with_copy))
    result = audit.audit_split(with_copy, both, "grouped", tmp_path)
    assert "identical-rows" in result.errors


def test_the_files_given_to_the_audit_have_no_ids_but_the_participant(tmp_path: Path) -> None:
    data = reviewed_of(participants=10, seed=1)
    split = cv.grouped_splits(data.y, data.groups, 2, 1, seed=0)[0]
    audit.audit_split(data.frame, split, "grouped", tmp_path)
    header = (tmp_path / "train.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "participant_code" in header and "qc" in header
    assert not {"recording_id", "session_id", "participant_id"} & set(header)
    assert set(audit.DROPPED) == {"recording_id", "session_id", "participant_id"}
    assert len((tmp_path / "test.csv").read_text(encoding="utf-8").splitlines()) == len(split.test) + 1


def test_without_table_audit_the_error_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "table_audit" else real(name, *a))
    with pytest.raises(audit.AuditUnavailable, match="(?s)not installed.*--skip-audit"):
        audit.check_available()
    data = reviewed_of(participants=10, seed=1)
    with pytest.raises(audit.AuditUnavailable):
        audit.audit_splits(data.frame, cv.grouped_splits(data.y, data.groups, 2, 1), "grouped")
