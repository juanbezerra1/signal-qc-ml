from __future__ import annotations

import numpy as np
import pytest

from signal_qc_ml import cv, evaluate, stats
from signal_qc_ml.data import Reviewed
from signal_qc_ml.models import FeatureSpec

from .helpers import reviewed_of


@pytest.fixture(scope="module")
def reviewed() -> Reviewed:
    return reviewed_of(participants=40, per_participant=8, seed=3)


def test_no_participant_is_on_both_sides_of_a_grouped_split(reviewed: Reviewed) -> None:
    splits = cv.grouped_splits(reviewed.y, reviewed.groups, 5, 3, seed=0)
    assert len(splits) == 15
    for split in splits:
        assert cv.shared_groups(split, reviewed.groups).size == 0


def test_every_recording_is_tested_once_in_every_pass(reviewed: Reviewed) -> None:
    n = len(reviewed.y)
    splits = cv.grouped_splits(reviewed.y, reviewed.groups, 5, 3, seed=0)
    for repeat in range(3):
        tested = np.concatenate([s.test for s in splits if s.repeat == repeat])
        assert sorted(tested) == list(range(n))
    for split in splits:
        assert np.intersect1d(split.train, split.test).size == 0
        assert split.train.size + split.test.size == n


def test_each_pass_cuts_the_participants_in_another_way(reviewed: Reviewed) -> None:
    splits = cv.grouped_splits(reviewed.y, reviewed.groups, 5, 2, seed=0)
    first = {tuple(s.test) for s in splits if s.repeat == 0}
    second = {tuple(s.test) for s in splits if s.repeat == 1}
    assert first != second


def test_the_same_seed_gives_the_same_splits(reviewed: Reviewed) -> None:
    a = cv.grouped_splits(reviewed.y, reviewed.groups, 5, 2, seed=7)
    b = cv.grouped_splits(reviewed.y, reviewed.groups, 5, 2, seed=7)
    c = cv.grouped_splits(reviewed.y, reviewed.groups, 5, 2, seed=8)
    assert all(np.array_equal(x.test, y.test) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x.test, y.test) for x, y in zip(a, c, strict=True))


def test_the_share_that_needs_attention_is_about_the_same_in_every_fold(reviewed: Reviewed) -> None:
    y = reviewed.y
    for split in cv.grouped_splits(y, reviewed.groups, 5, 2, seed=0):
        assert abs(y[split.test].mean() - y.mean()) < 0.1


def test_a_split_that_ignores_participants_shares_them(reviewed: Reviewed) -> None:
    splits = cv.random_splits(reviewed.y, 5, 1, seed=0)
    assert all(cv.shared_groups(s, reviewed.groups).size > 0 for s in splits)
    with pytest.raises(cv.LeakageError, match="participants are on both sides of fold 0 of pass 0"):
        cv.assert_no_shared_groups(splits, reviewed.groups)


def test_predictions_cover_every_recording_in_every_pass(reviewed: Reviewed) -> None:
    columns = list(reviewed.frame.columns)
    spec = evaluate.before_review(columns)
    splits = cv.grouped_splits(reviewed.y, reviewed.groups, 4, 2, seed=0)
    predictions = cv.out_of_fold("logistic", spec, reviewed.frame, reviewed.y, splits)
    assert predictions.score.shape == predictions.fold.shape == (2, len(reviewed.y))
    assert not np.isnan(predictions.score).any()
    assert set(np.unique(predictions.fold)) == {0, 1, 2, 3}
    assert ((predictions.score >= 0) & (predictions.score <= 1)).all()


def test_the_scores_are_the_same_with_one_process_or_two(reviewed: Reviewed) -> None:
    columns = list(reviewed.frame.columns)
    spec = evaluate.before_review(columns)
    splits = cv.grouped_splits(reviewed.y, reviewed.groups, 3, 2, seed=0)
    one = cv.out_of_fold("logistic", spec, reviewed.frame, reviewed.y, splits, jobs=1)
    two = cv.out_of_fold("logistic", spec, reviewed.frame, reviewed.y, splits, jobs=2)
    assert np.array_equal(one.score, two.score)


def test_the_prior_is_exactly_at_chance_whatever_the_folds(reviewed: Reviewed) -> None:
    splits = cv.grouped_splits(reviewed.y, reviewed.groups, 5, 3, seed=0)
    predictions = cv.out_of_fold("prior", FeatureSpec("nothing"), reviewed.frame, reviewed.y, splits)
    assert stats.FoldedAUC(predictions.score, predictions.fold).auc(reviewed.y) == 0.5


def test_the_number_of_recordings_that_need_attention_is_moved_to_participants_of_the_same_size() -> None:
    rng = np.random.default_rng(0)
    sizes = [8] * 20 + [5] * 6 + [3]  # one participant is the only one with three recordings
    groups = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    y = (rng.random(groups.size) < 0.3).astype(np.int8)
    moved = cv.permute_between(y, groups, np.random.default_rng(1))
    before = {g: int(y[groups == g].sum()) for g in np.unique(groups)}
    after = {g: int(moved[groups == g].sum()) for g in np.unique(groups)}
    for size in (8, 5):
        members = [g for g in before if sizes[g] == size]
        assert sorted(before[g] for g in members) == sorted(after[g] for g in members)  # the same counts, other owners
        assert [before[g] for g in members] != [after[g] for g in members]
    assert after[26] == before[26]  # nobody else has three recordings: it keeps its own count
    assert moved.sum() == y.sum()
    assert not np.array_equal(moved, y)


def test_moving_does_not_carry_the_order_of_the_recordings_along() -> None:
    """If a participant's results were moved as they are, the position of a recording would still say its result."""
    groups = np.repeat(np.arange(200), 8)
    position = np.tile(np.arange(8), 200)
    y = (position == 0).astype(np.int8)  # in every participant the first recording is the one that needs attention
    assert np.corrcoef(position == 0, y)[0, 1] == pytest.approx(1.0)
    moved = cv.permute_between(y, groups, np.random.default_rng(0))
    assert np.array_equal(np.bincount(groups, weights=moved), np.ones(200))  # one each, as before
    assert abs(np.corrcoef(position == 0, moved)[0, 1]) < 0.15  # but not the first one any more


def test_moving_results_breaks_the_link_with_the_columns() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 8)
    y = (rng.random(groups.size) < 0.3).astype(np.int8)
    feature = y + rng.normal(size=y.size)
    assert np.corrcoef(feature, y)[0, 1] > 0.4
    moved = cv.permute_between(y, groups, np.random.default_rng(2))
    assert abs(np.corrcoef(feature, moved)[0, 1]) < 0.1


def test_moving_results_keeps_how_alike_a_participants_recordings_are() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 8)
    chance = rng.beta(0.3, 1.7, size=200)
    y = (rng.random(groups.size) < chance[groups]).astype(np.int8)
    moved = cv.permute_between(y, groups, np.random.default_rng(5))

    def spread(values: np.ndarray) -> float:
        return float(np.var([values[groups == g].mean() for g in range(200)]))

    assert spread(moved) == pytest.approx(spread(y), rel=1e-9)  # participants have the same means, only other owners
    plain = np.random.default_rng(5).permutation(y)
    assert spread(plain) < 0.5 * spread(y)  # a plain shuffle would have lost it


def test_moving_is_deterministic_given_the_generator() -> None:
    groups = np.repeat(np.arange(50), 8)
    y = (np.random.default_rng(0).random(groups.size) < 0.3).astype(np.int8)
    a = cv.permute_between(y, groups, np.random.default_rng(9))
    b = cv.permute_between(y, groups, np.random.default_rng(9))
    assert np.array_equal(a, b)
    assert a is not y and a.dtype == y.dtype


def test_the_same_seed_puts_the_same_participants_in_each_fold_in_every_version_that_is_allowed() -> None:
    """``results/`` holds numbers that depend on which participants share a fold. ``StratifiedGroupKFold`` cut them in
    another way up to scikit-learn 1.7 (same seed, other participants), which is why ``pyproject.toml`` asks for 1.8.
    If a later version changes it again, this is the test that says so, before the tests of the stored numbers do."""
    groups = np.repeat(np.arange(20), 4)
    rng = np.random.default_rng(7)
    chance = rng.beta(0.6, 3.4, size=20)  # some participants have many recordings to look at, most have none
    y = (rng.random(groups.size) < chance[groups]).astype(np.int8)
    splits = cv.grouped_splits(y, groups, folds=4, repeats=1, seed=0)
    assert [sorted(set(groups[s.test].tolist())) for s in splits] == [
        [0, 2, 7, 10, 15, 18],
        [1, 12, 13, 16, 17],
        [4, 9, 11, 19],
        [3, 5, 6, 8, 14],
    ]
