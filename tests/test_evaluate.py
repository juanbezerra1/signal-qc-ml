from __future__ import annotations

import numpy as np
import pytest

from signal_qc_ml import columns as cols
from signal_qc_ml import cv, evaluate

from .helpers import make_table, reviewed_of

QUICK = evaluate.Design(folds=4, repeats=2, seed=0, bootstrap=60, permutations=40)


@pytest.fixture(scope="module")
def noise() -> evaluate.Evaluation:
    return evaluate.evaluate(reviewed_of(participants=60, seed=1), QUICK)


def test_no_set_of_columns_reads_the_target_an_identifier_or_the_review() -> None:
    columns = list(make_table().columns)
    specs = [*evaluate.step_specs(columns), evaluate.before_review(columns)]
    forbidden = {cols.TARGET, *cols.IDENTIFIERS, *cols.REVIEW.columns}
    for spec in specs:
        assert forbidden.isdisjoint(spec.columns), spec.name


def test_the_review_columns_are_only_in_the_rows_that_say_so(noise: evaluate.Evaluation) -> None:
    for row in noise.rows:
        assert row.known_before_review == (row.features not in (evaluate.REVIEW_ONLY_NAME, evaluate.BOTH_NAME)), (
            row.name
        )


def test_the_groups_of_columns_are_added_one_at_a_time() -> None:
    columns = list(make_table().columns)
    specs = evaluate.step_specs(columns)
    assert [s.name for s in specs] == ["acquisition", "+ session", "+ device", "+ participant", "+ signal"]
    sizes = [len(s) for s in specs]
    assert sizes == sorted(set(sizes))  # every step adds columns
    for before, after in zip(specs, specs[1:], strict=False):
        assert set(before.columns) < set(after.columns)
    assert set(specs[-1].columns) == set(evaluate.before_review(columns).columns)


def test_a_table_without_signal_columns_has_one_step_fewer() -> None:
    frame = make_table().drop(columns=list(cols.signal_columns(list(make_table().columns))))
    assert [s.name for s in evaluate.step_specs(list(frame.columns))][-1] == "+ participant"


def test_the_prior_is_at_one_half_whatever_the_data(noise: evaluate.Evaluation) -> None:
    prior = noise.row("nothing", "prior")
    assert (prior.auc, prior.ci_low, prior.ci_high) == (0.5, 0.5, 0.5)
    assert prior.n_features == 0


def test_columns_known_before_the_review_do_not_beat_chance_when_nothing_is_there(noise: evaluate.Evaluation) -> None:
    for row in noise.rows:
        if row.known_before_review and row.model != "prior":
            assert 0.35 < row.auc < 0.65, row.name
            assert row.interval_has_chance, row.name


def test_the_columns_that_the_review_produces_give_the_answer_away(noise: evaluate.Evaluation) -> None:
    assert noise.row(evaluate.REVIEW_ONLY_NAME).auc > 0.95
    assert noise.row(evaluate.BOTH_NAME, "boosting").auc > 0.95


def test_the_interval_holds_the_estimate(noise: evaluate.Evaluation) -> None:
    for row in noise.rows:
        assert row.ci_low <= row.auc <= row.ci_high, row.name


def test_the_primary_test_is_on_the_columns_known_before_the_review(noise: evaluate.Evaluation) -> None:
    assert noise.primary is not None
    assert noise.primary.permutations == QUICK.permutations
    assert len(noise.primary.null) == QUICK.permutations
    assert noise.primary.p_value > 0.05  # nothing is there
    assert 0.4 < noise.primary.null_mean < 0.6
    assert noise.primary.critical_95 >= noise.primary.null_mean


def test_the_primary_test_sees_an_effect_that_is_planted() -> None:
    result = evaluate.evaluate(reviewed_of(participants=60, effect=1.5, seed=1), QUICK, boosting=False)
    assert result.primary is not None
    assert result.primary.p_value < 0.05
    assert result.primary.auc > result.primary.critical_95
    assert result.row("+ signal").auc > 0.6 and not result.row("+ signal").interval_has_chance


def test_the_evaluation_is_the_same_every_time_and_with_more_processes() -> None:
    data = reviewed_of(participants=40, seed=2)
    small = evaluate.Design(folds=3, repeats=2, bootstrap=30, permutations=12)
    a = evaluate.evaluate(data, small, jobs=1, boosting=False)
    b = evaluate.evaluate(data, small, jobs=1, boosting=False)
    c = evaluate.evaluate(data, small, jobs=2, boosting=False)
    assert [r.as_dict() for r in a.rows] == [r.as_dict() for r in b.rows] == [r.as_dict() for r in c.rows]
    assert a.primary is not None and c.primary is not None
    assert a.primary.as_dict() == c.primary.as_dict()


def test_a_different_seed_gives_other_numbers() -> None:
    data = reviewed_of(participants=40, seed=2)
    one = evaluate.evaluate(data, evaluate.Design(3, 2, 0, 30, 0), boosting=False)
    two = evaluate.evaluate(data, evaluate.Design(3, 2, 1, 30, 0), boosting=False)
    assert one.row("acquisition").auc != two.row("acquisition").auc


def test_without_permutations_there_is_no_primary_test() -> None:
    result = evaluate.evaluate(reviewed_of(participants=30), evaluate.Design(3, 1, 0, 20, 0), boosting=False)
    assert result.primary is None


def test_a_row_says_what_it_is_in_the_json(noise: evaluate.Evaluation) -> None:
    as_dict = noise.rows[1].as_dict()
    assert set(as_dict) == {
        "name", "model", "features", "n_features", "auc", "auc_ci95", "average_precision", "brier", "known_before_review",
    }  # fmt: skip
    assert as_dict["auc_ci95"] == [noise.rows[1].ci_low, noise.rows[1].ci_high]


def test_the_row_that_is_asked_for_has_to_exist(noise: evaluate.Evaluation) -> None:
    with pytest.raises(KeyError, match="no row for 'weather'"):
        noise.row("weather")


def test_the_interval_has_chance_unless_it_is_on_one_side_of_one_half() -> None:
    def row(low: float, high: float) -> evaluate.Row:
        return evaluate.Row("n", "m", "f", 1, (low + high) / 2, low, high, 0.1, 0.1, True)

    assert row(0.45, 0.55).interval_has_chance
    assert row(0.5, 0.6).interval_has_chance
    assert not row(0.51, 0.6).interval_has_chance
    assert not row(0.4, 0.49).interval_has_chance


def test_the_statistic_of_the_test_is_the_first_pass_of_the_main_evaluation() -> None:
    data = reviewed_of(participants=40, seed=2)
    plan = evaluate.Design(folds=3, repeats=1, seed=4, bootstrap=20, permutations=0)
    result = evaluate.evaluate(data, plan, boosting=False)
    known = evaluate.before_review(list(data.frame.columns))
    alone = evaluate.single_pass_auc("logistic", known, data.frame, data.y, data.groups, 3, 4)
    assert alone == pytest.approx(result.row("+ signal").auc, abs=1e-12)


def test_the_moved_tables_are_scored_by_the_same_procedure() -> None:
    data = reviewed_of(participants=40, seed=2)
    known = evaluate.before_review(list(data.frame.columns))
    moved = cv.permute_between(data.y, data.groups, np.random.default_rng(3))
    value = evaluate.single_pass_auc("logistic", known, data.frame, moved, data.groups, 3, 0)
    assert 0.2 < value < 0.8


def test_the_primary_test_is_of_the_logistic_model_on_everything_known_before_the_review() -> None:
    data = reviewed_of(participants=40, seed=2)
    plan = evaluate.Design(folds=3, repeats=1, seed=1, bootstrap=20, permutations=4)
    columns = list(data.frame.columns)
    primary = evaluate.permutation_test(evaluate.before_review(columns), data, plan)
    assert evaluate.PRIMARY_MODEL == "logistic"
    expected = evaluate.single_pass_auc(
        "logistic", evaluate.before_review(columns), data.frame, data.y, data.groups, 3, 1
    )
    assert primary.auc == pytest.approx(expected, abs=1e-12)
    assert primary.as_dict()["model"] == "logistic"
