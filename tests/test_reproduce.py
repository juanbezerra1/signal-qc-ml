"""The numbers in ``results/`` can be made again: one row of the table, with the design that was used for all of them."""

from __future__ import annotations

import json

import pytest

from signal_qc_ml import columns as cols
from signal_qc_ml import cv
from signal_qc_ml.data import read_table, reviewed_and_pending
from signal_qc_ml.evaluate import Design, evaluate_one, step_specs

from .helpers import RESULTS, snapshot_path

# Another machine may have another BLAS, and the last digits of a logistic regression follow it. A change in what the
# code does moves the AUC by far more than this.
TOLERANCE = 1e-3


def test_a_row_of_the_small_results_is_made_again_by_the_code() -> None:
    stored = json.loads((RESULTS / "small.json").read_text(encoding="utf-8"))
    design = Design(**stored["design"])
    data, _ = reviewed_and_pending(read_table(snapshot_path("small")))
    splits = cv.grouped_splits(data.y, data.groups, design.folds, design.repeats, design.seed)
    spec = step_specs(list(data.frame.columns))[0]
    assert spec.name == cols.STEPS[0]

    row, _ = evaluate_one("logistic", spec, data, splits, design)

    expected = next(r for r in stored["rows"] if r["model"] == "logistic" and r["features"] == spec.name)
    assert row.n_features == expected["n_features"]
    assert row.auc == pytest.approx(expected["auc"], abs=TOLERANCE)
    assert row.ci_low == pytest.approx(expected["auc_ci95"][0], abs=TOLERANCE)
    assert row.ci_high == pytest.approx(expected["auc_ci95"][1], abs=TOLERANCE)
    assert row.average_precision == pytest.approx(expected["average_precision"], abs=TOLERANCE)
    assert row.brier == pytest.approx(expected["brier"], abs=TOLERANCE)
