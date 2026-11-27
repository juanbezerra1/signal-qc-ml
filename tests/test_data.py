from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from signal_qc_ml import columns as cols
from signal_qc_ml import data

from .helpers import make_table, write_csv


def test_reading_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(data.TableError, match="there is no file"):
        data.read_table(tmp_path / "nothing.csv")


def test_reading_something_that_is_not_a_table(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(data.TableError, match="not a table of recordings"):
        data.read_table(path)


def test_a_missing_column_is_named(tmp_path: Path) -> None:
    frame = make_table().drop(columns=["room", "qc"])
    path = write_csv(frame, tmp_path / "t.csv")
    with pytest.raises(data.TableError, match="no column .*room.*qc"):
        data.read_table(path)


def test_an_unknown_state_is_named() -> None:
    frame = make_table()
    frame.loc[0, "qc"] = "maybe"
    with pytest.raises(data.TableError, match="not one of .*maybe"):
        data.validate(frame)


def test_a_repeated_recording_is_refused() -> None:
    frame = make_table()
    ids = frame["recording_id"].to_numpy().copy()
    ids[1] = ids[0]
    frame["recording_id"] = ids
    with pytest.raises(data.TableError, match="more than once"):
        data.validate(frame)


def test_a_signal_column_must_be_numbers() -> None:
    frame = make_table()
    frame["sig_alpha_mean"] = "x"
    with pytest.raises(data.TableError, match="sig_alpha_mean is not made of numbers"):
        data.validate(frame)


def test_a_blank_cell_is_missing_and_not_the_word_na(tmp_path: Path) -> None:
    frame = make_table()
    path = write_csv(frame, tmp_path / "t.csv")
    back = data.read_table(path)
    assert back["sig_gamma_mean"].isna().sum() == frame["sig_gamma_mean"].isna().sum()


def test_pending_is_not_a_class_of_the_target() -> None:
    frame = make_table(pending=5)
    reviewed, pending = data.reviewed_and_pending(frame)
    assert len(pending) == 5 and set(pending["qc"]) == {cols.PENDING}
    assert len(reviewed.frame) + len(pending) == len(frame)
    assert cols.PENDING not in set(reviewed.frame["qc"])


def test_flagged_and_rejected_are_the_class_that_needs_attention() -> None:
    qc = pd.Series(["passed", "flagged", "rejected", "passed"])
    assert data.target_of(qc).tolist() == [0, 1, 1, 0]


def test_the_target_and_the_groups_line_up_with_the_rows() -> None:
    reviewed, _ = data.reviewed_and_pending(make_table())
    assert len(reviewed.y) == len(reviewed.groups) == len(reviewed.frame)
    assert reviewed.y.tolist() == reviewed.frame["qc"].isin(["flagged", "rejected"]).astype(int).tolist()
    assert reviewed.groups.tolist() == reviewed.frame["participant_id"].tolist()


def test_the_numbers_that_every_report_starts_with() -> None:
    frame = make_table(participants=10, per_participant=8, pending=4)
    numbers = data.counts(frame)
    assert numbers["recordings"] == 80 and numbers["pending"] == 4 and numbers["reviewed"] == 76
    assert numbers["participants"] == 10
    assert numbers["positives"] == int(frame["qc"].isin(["flagged", "rejected"]).sum())


def test_nothing_reviewed_is_an_error() -> None:
    frame = make_table()
    frame["qc"] = cols.PENDING
    with pytest.raises(data.TableError, match="nothing to learn from"):
        data.reviewed_and_pending(frame)


def test_a_single_result_is_an_error() -> None:
    frame = make_table()
    frame["qc"] = "passed"
    with pytest.raises(data.TableError, match="same result"):
        data.reviewed_and_pending(frame)
