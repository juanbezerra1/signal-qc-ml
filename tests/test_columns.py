from __future__ import annotations

import pytest

from signal_qc_ml import columns as cols
from signal_qc_ml.data import read_table

from .helpers import make_table, snapshot_path


def test_no_column_is_in_two_groups() -> None:
    seen: list[str] = []
    for group in (*cols.BEFORE_REVIEW, cols.REVIEW):
        seen.extend(group.columns)
    assert len(seen) == len(set(seen))


def test_identifiers_and_target_are_never_features() -> None:
    features = {c for group in (*cols.BEFORE_REVIEW, cols.REVIEW) for c in group.columns}
    assert features.isdisjoint(cols.IDENTIFIERS)
    assert cols.TARGET not in features


def test_the_review_columns_are_not_known_before_the_review() -> None:
    known = {c for group in cols.BEFORE_REVIEW for c in group.columns}
    assert set(cols.REVIEW.columns) == {"n_trials_rejected", "valid_fraction"}
    assert known.isdisjoint(cols.REVIEW.columns)


def test_steps_are_the_four_groups_and_the_signal() -> None:
    assert (*(g.name for g in cols.BEFORE_REVIEW), "signal") == cols.STEPS


def test_the_group_of_participants_is_a_column_of_the_table() -> None:
    assert cols.GROUP in cols.IDENTIFIERS and cols.GROUP_LABEL in cols.IDENTIFIERS


def test_signal_columns_are_found_by_prefix_in_table_order() -> None:
    names = ["a", "sig_x_mean", "b", "sig_x_sd", "sig_y_mean"]
    assert cols.signal_columns(names) == ("sig_x_mean", "sig_x_sd", "sig_y_mean")
    assert cols.group_named("signal", names).numeric == ("sig_x_mean", "sig_x_sd", "sig_y_mean")


def test_an_unknown_group_says_which_ones_there_are() -> None:
    with pytest.raises(KeyError, match="acquisition, session, device, participant, signal, review"):
        cols.group_named("weather", [])


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_the_real_tables_have_exactly_the_columns_that_the_code_knows(size: str) -> None:
    frame = read_table(snapshot_path(size))
    expected = [*cols.required_columns(), *cols.signal_columns(list(frame.columns))]
    assert sorted(frame.columns) == sorted(expected)
    assert cols.signal_columns(list(frame.columns)), "there must be signal columns"


def test_a_synthetic_table_has_the_same_columns_as_the_real_ones() -> None:
    real = read_table(snapshot_path("small"))
    fake = make_table()
    assert set(cols.required_columns()) <= set(fake.columns)
    assert set(cols.required_columns()) <= set(real.columns)
