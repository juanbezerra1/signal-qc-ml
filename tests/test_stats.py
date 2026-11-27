from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from signal_qc_ml import stats


def labels_and_scores(n: int = 300, seed: int = 0, ties: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    score = rng.normal(size=n) + 0.7 * y
    return y, (np.round(score, 1) if ties else score)


@pytest.mark.parametrize("ties", [False, True])
def test_auc_is_the_one_of_scikit_learn(ties: bool) -> None:
    for seed in range(5):
        y, score = labels_and_scores(seed=seed, ties=ties)
        assert stats.auc(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_a_tie_counts_as_half() -> None:
    assert stats.auc(np.array([0, 1]), np.array([1.0, 1.0])) == 0.5
    # two pairs are won (0.9 against 0.2) and two are tied (0.2 against 0.2): (2 + 2 * 0.5) / 4
    assert stats.auc(np.array([0, 1, 0, 1]), np.array([0.2, 0.9, 0.2, 0.2])) == 0.75


def test_a_perfect_and_a_reversed_ranking() -> None:
    y = np.array([0, 0, 1, 1])
    assert stats.auc(y, np.array([1, 2, 3, 4])) == 1.0
    assert stats.auc(y, np.array([4, 3, 2, 1])) == 0.0


def test_weights_are_the_same_as_repeating_the_rows() -> None:
    y, score = labels_and_scores(n=60, ties=True)
    weights = np.random.default_rng(3).integers(0, 4, size=y.size).astype(float)
    repeated = np.repeat(np.arange(y.size), weights.astype(int))
    expected = roc_auc_score(y[repeated], score[repeated])
    assert stats.RankedScores(score).auc(y, weights) == pytest.approx(expected, abs=1e-12)


def test_one_class_has_no_auc() -> None:
    assert math.isnan(stats.auc(np.zeros(5), np.arange(5.0)))
    y, score = labels_and_scores(n=40)
    weights = np.where(y == 1, 0.0, 1.0)  # no recording that needs attention is left after the weighting
    assert math.isnan(stats.RankedScores(score).auc(y, weights))


def test_the_auc_of_folds_is_the_mean_of_the_aucs_of_each_fold() -> None:
    y, score = labels_and_scores(n=200)
    fold = np.arange(y.size) % 4
    expected = np.mean([roc_auc_score(y[fold == k], score[fold == k]) for k in range(4)])
    assert stats.FoldedAUC(score, fold).auc(y) == pytest.approx(expected, abs=1e-12)


def test_the_auc_of_folds_does_not_rank_one_fold_against_another() -> None:
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])  # the second fold has higher scores than the first one: nothing to compare
    assert stats.FoldedAUC(score, np.array([0, 0, 1, 1])).auc(y) == 1.0
    assert stats.auc(y, score) == 0.75


def test_a_model_that_says_the_same_to_every_recording_is_at_one_half_in_every_fold() -> None:
    y, _ = labels_and_scores(n=200)
    fold = np.arange(y.size) % 5
    score = np.where(fold < 2, 0.3, 0.1)  # a constant in each fold, different from fold to fold: what a prior gives
    assert stats.FoldedAUC(score, fold).auc(y) == 0.5


def test_a_fold_without_both_classes_is_left_out() -> None:
    y = np.array([0, 0, 0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.9])
    fold = np.array([0, 0, 0, 1, 1, 1])
    assert stats.FoldedAUC(score, fold).auc(y) == roc_auc_score(y[3:], score[3:])


def test_folded_scores_and_folds_must_have_the_same_shape() -> None:
    with pytest.raises(ValueError, match="same shape"):
        stats.FoldedAUC(np.zeros((2, 5)), np.zeros((1, 5)))


def test_several_passes_are_averaged() -> None:
    y, score = labels_and_scores(n=100)
    other = np.random.default_rng(1).permutation(score)
    folds = np.arange(y.size) % 5
    both = stats.FoldedAUC(np.vstack([score, other]), np.vstack([folds, folds])).auc(y)
    first = stats.FoldedAUC(score, folds).auc(y)
    second = stats.FoldedAUC(other, folds).auc(y)
    assert both == pytest.approx((first + second) / 2)


def groups_of(n_groups: int, size: int) -> np.ndarray:
    return np.repeat(np.arange(n_groups), size)


def test_the_interval_is_reproducible_and_holds_the_estimate() -> None:
    groups = groups_of(40, 5)
    y, score = labels_and_scores(n=groups.size, seed=4)
    folded = stats.FoldedAUC(score, np.arange(y.size) % 5)
    low, high = stats.participant_bootstrap(y, folded, groups, 300, seed=1)
    assert (low, high) == stats.participant_bootstrap(y, folded, groups, 300, seed=1)
    assert low < folded.auc(y) < high
    assert (low, high) != stats.participant_bootstrap(y, folded, groups, 300, seed=2)


def test_the_interval_is_wider_with_fewer_participants() -> None:
    widths = []
    for n_groups in (30, 300):
        groups = groups_of(n_groups, 5)
        y, score = labels_and_scores(n=groups.size, seed=n_groups)
        folded = stats.FoldedAUC(score, np.arange(y.size) % 5)
        low, high = stats.participant_bootstrap(y, folded, groups, 400, seed=0)
        widths.append(high - low)
    assert widths[0] > 2 * widths[1]


def test_a_perfect_ranking_has_no_uncertainty() -> None:
    groups = groups_of(30, 4)
    y = (np.arange(groups.size) % 4 == 0).astype(np.int8)
    folded = stats.FoldedAUC(y.astype(float), np.arange(y.size) % 5)
    assert stats.participant_bootstrap(y, folded, groups, 100, seed=0) == (1.0, 1.0)


def test_the_interval_covers_the_truth_about_as_often_as_it_says() -> None:
    """Scores with a known AUC (a shift of one standard deviation: Phi(1/sqrt 2) = 0.76), many samples, 95% intervals."""
    truth = stats.theory_auc(1.0)
    hits = 0
    runs = 120
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        groups = groups_of(60, 5)
        y = (rng.random(groups.size) < 0.25).astype(np.int8)
        score = rng.normal(size=groups.size) + y
        folded = stats.FoldedAUC(score, np.arange(y.size) % 5)
        low, high = stats.participant_bootstrap(y, folded, groups, 200, seed)
        hits += low <= truth <= high
    assert 0.88 <= hits / runs <= 1.0  # 95% in theory; 120 runs allow a few percent either way


def test_the_theory_of_a_shift() -> None:
    assert stats.theory_auc(0.0) == 0.5
    assert stats.theory_auc(1.0) == pytest.approx(0.7602, abs=1e-4)
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.5).astype(np.int8)
    assert stats.auc(y, rng.normal(size=y.size) + 0.5 * y) == pytest.approx(stats.theory_auc(0.5), abs=0.004)


def test_the_p_value_counts_the_observed_value_too() -> None:
    null = np.arange(10.0)
    assert stats.p_value(100.0, null) == 1 / 11  # nothing in 10 draws is as large: not zero, one in eleven
    assert stats.p_value(-1.0, null) == 1.0
    assert stats.p_value(5.0, null) == (1 + 5) / 11  # the draws 5 to 9 are as large


def test_the_interval_is_made_of_the_two_and_a_half_and_the_ninety_seven_and_a_half_percentiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[float]] = []
    real = np.nanpercentile

    def spy(values: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(list(q))
        return real(values, q, *args, **kwargs)

    monkeypatch.setattr(np, "nanpercentile", spy)
    groups = groups_of(20, 5)
    y, score = labels_and_scores(n=groups.size, seed=4)
    stats.participant_bootstrap(y, stats.FoldedAUC(score, np.arange(y.size) % 5), groups, 50, seed=0)
    assert seen == [[2.5, 97.5]]
