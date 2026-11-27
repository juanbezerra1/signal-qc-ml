from __future__ import annotations

import numpy as np
import pytest

from signal_qc_ml import controls, evaluate, stats
from signal_qc_ml.data import Reviewed

from .helpers import reviewed_of

PLAN = evaluate.Design(folds=4, repeats=1, seed=0, bootstrap=20, permutations=0)


def test_the_planted_column_moves_by_the_shift_for_the_recordings_that_need_attention() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.3).astype(np.int8)
    column = controls.planted_column(y, 0.5, rng)
    assert column[y == 1].mean() - column[y == 0].mean() == pytest.approx(0.5, abs=0.03)
    assert column[y == 0].std() == pytest.approx(1.0, abs=0.02)


def test_a_planted_column_alone_gives_the_auc_that_the_theory_says() -> None:
    data = reviewed_of(participants=150, seed=3)
    points = controls.power_curve(data, critical=0.6, design=PLAN, shifts=(0.0, 0.5, 1.0), replicates=6)
    for point in points:
        assert point.auc_alone == pytest.approx(point.auc_theory, abs=0.04), point
        assert point.auc_theory == stats.theory_auc(point.shift)


def test_a_stronger_effect_is_seen_more_often() -> None:
    data = reviewed_of(participants=150, seed=3)
    points = controls.power_curve(data, critical=0.58, design=PLAN, shifts=(0.0, 1.0, 2.0), replicates=6)
    seen = [p.detected for p in points]
    assert seen[0] <= 0.34 and seen[-1] == 1.0 and seen == sorted(seen)
    assert points[-1].auc_with_known > points[0].auc_with_known


def test_the_planted_curve_does_not_see_what_the_real_table_already_knows() -> None:
    """The results are moved before the column is planted, so a column of the table that really carries the result
    (here, one that moves by 3 standard deviations) must not help: with no planted effect there is nothing to see."""
    data = reviewed_of(participants=150, effect=3.0, seed=3)
    (nothing,) = controls.power_curve(data, critical=0.6, design=PLAN, shifts=(0.0,), replicates=6)
    assert nothing.auc_with_known < 0.6 and nothing.detected == 0.0


def test_the_planted_curve_is_the_same_every_time() -> None:
    data = reviewed_of(participants=60, seed=3)
    one = controls.power_curve(data, 0.6, PLAN, shifts=(0.0, 1.0), replicates=3)
    two = controls.power_curve(data, 0.6, PLAN, shifts=(0.0, 1.0), replicates=3, jobs=2)
    assert one == two


def test_a_table_that_already_has_the_planted_column_is_refused() -> None:
    data = reviewed_of(participants=30)
    data = Reviewed(data.frame.assign(**{controls.PLANTED: 0.0}), data.y, data.groups)
    with pytest.raises(ValueError, match="already has a column called planted"):
        controls.power_curve(data, 0.6, PLAN, shifts=(0.0,), replicates=1)


def point(shift: float, seen: float) -> controls.PowerPoint:
    return controls.PowerPoint(shift, 0.5, 0.5, 0.5, seen, 10)


def test_the_smallest_effect_that_is_seen_is_the_first_one_that_reaches_the_power() -> None:
    points = [point(0.0, 0.9), point(0.2, 0.3), point(0.4, 0.8), point(0.6, 1.0)]
    found = controls.smallest_detected(points)
    assert found is not None and found.shift == 0.4  # not the zero effect, even if it was "seen" by chance
    assert controls.smallest_detected([point(0.0, 0.0), point(0.5, 0.79)]) is None


def test_a_tendency_of_each_participant_makes_their_recordings_alike() -> None:
    groups = np.repeat(np.arange(400), 8)
    y = controls.simulate_tendency(groups, 0.12, 3.3, np.random.default_rng(0))
    assert y.mean() == pytest.approx(0.12, abs=0.02)
    means = np.array([y[groups == g].mean() for g in range(400)])
    binomial = 0.12 * 0.88 / 8
    assert means.var() > 2 * binomial  # participants differ much more than chance alone would make them
    same = controls.simulate_tendency(groups, 0.12, 3.3, np.random.default_rng(0))
    assert np.array_equal(y, same)
    assert not np.array_equal(y, controls.simulate_tendency(groups, 0.12, 3.3, np.random.default_rng(1)))


def test_a_large_concentration_makes_participants_alike() -> None:
    groups = np.repeat(np.arange(400), 8)
    spread = []
    for concentration in (1.0, 1000.0):
        y = controls.simulate_tendency(groups, 0.2, concentration, np.random.default_rng(0))
        spread.append(np.var([y[groups == g].mean() for g in range(400)]))
    assert spread[0] > 3 * spread[1]


def test_a_random_split_reports_what_a_split_by_participant_does_not() -> None:
    data = reviewed_of(participants=120, seed=3)
    leak = controls.identity_leak(data, PLAN, concentration=2.0, simulations=4)
    assert leak.auc_random > leak.auc_grouped + 0.05
    assert abs(leak.auc_grouped - 0.5) < 0.1
    assert leak.random_higher >= 3
    assert leak.icc == pytest.approx(1 / 3.0)
    assert leak.simulations == 4
    assert leak.as_dict()["concentration"] == 2.0


def test_the_columns_of_the_table_are_left_out_of_the_comparison() -> None:
    """A table whose own columns carry a signal still gives a planted effect of zero nothing to see."""
    data = reviewed_of(participants=150, effect=1.5, seed=3)
    with_signal = evaluate.evaluate(data, evaluate.Design(4, 1, 0, 20, 0), boosting=False).row("+ signal").auc
    assert with_signal > 0.65  # the table does carry a signal
    (zero,) = controls.power_curve(data, critical=0.6, design=PLAN, shifts=(0.0,), replicates=4)
    assert zero.auc_with_known < 0.58  # but the results were moved first, so it is not what the curve measures
