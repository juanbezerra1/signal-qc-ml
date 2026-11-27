from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from signal_qc_ml import columns as cols
from signal_qc_ml import models
from signal_qc_ml.data import Reviewed

from .helpers import make_table, reviewed_of


@pytest.fixture(scope="module")
def reviewed() -> Reviewed:
    return reviewed_of(participants=30, per_participant=8, seed=5)


def fitted(model: str, reviewed: Reviewed, steps: list[str] | None = None) -> tuple[Any, models.FeatureSpec]:
    frame = reviewed.frame
    spec = models.spec_for(steps or cols.STEPS, list(frame.columns))
    pipeline = models.build_pipeline(model, spec, seed=0)
    pipeline.fit(models.design(frame, spec), reviewed.y)
    return pipeline, spec


def test_the_prior_says_the_share_seen_in_training(reviewed: Reviewed) -> None:
    pipeline, spec = fitted("prior", reviewed, [])
    proba = pipeline.predict_proba(models.design(reviewed.frame, spec))[:, 1]
    assert np.allclose(proba, reviewed.y.mean())


def test_the_prior_reads_nothing() -> None:
    frame = make_table()
    assert list(models.design(frame, models.FeatureSpec("nothing")).columns) == ["nothing"]


@pytest.mark.parametrize("model", ["logistic", "boosting"])
def test_a_model_scores_every_recording_with_a_probability(model: str, reviewed: Reviewed) -> None:
    pipeline, spec = fitted(model, reviewed)
    proba = pipeline.predict_proba(models.design(reviewed.frame, spec))[:, 1]
    assert proba.shape == (len(reviewed.y),) and ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("model", ["logistic", "boosting"])
def test_a_code_never_seen_in_training_does_not_stop_the_model(model: str, reviewed: Reviewed) -> None:
    pipeline, spec = fitted(model, reviewed)
    new = reviewed.frame.head(5).copy()
    new["room"] = "R99"
    new["device_id"] = 9999
    proba = pipeline.predict_proba(models.design(new, spec))[:, 1]
    assert not np.isnan(proba).any()


@pytest.mark.parametrize("model", ["logistic", "boosting"])
def test_a_column_that_a_modality_does_not_have_is_not_an_error(model: str, reviewed: Reviewed) -> None:
    frame = reviewed.frame
    assert frame["sig_gamma_mean"].isna().any() and frame["sig_gamma_mean"].notna().any()
    pipeline, spec = fitted(model, reviewed)
    assert not np.isnan(pipeline.predict_proba(models.design(frame, spec))).any()


def test_codes_are_categories_and_not_quantities(reviewed: Reviewed) -> None:
    frame = reviewed.frame
    spec = models.spec_for(["device"], list(frame.columns))
    assert "device_id" in spec.categorical and "device_id" not in spec.numeric
    assert all(isinstance(v, str) for v in models.design(frame, spec)["device_id"])


def test_an_unknown_model_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown model 'forest'.*prior, logistic, boosting"):
        models.build_pipeline("forest", models.FeatureSpec("x", ("a",)))


def test_a_spec_for_the_review_has_only_the_two_review_columns(reviewed: Reviewed) -> None:
    spec = models.spec_for(["review"], list(reviewed.frame.columns))
    assert set(spec.columns) == {"n_trials_rejected", "valid_fraction"}


def test_a_spec_that_adds_a_column_keeps_the_rest() -> None:
    spec = models.FeatureSpec("a", ("x",), ("c",)).plus("y", name="b")
    assert (spec.name, spec.numeric, spec.categorical, len(spec)) == ("b", ("x", "y"), ("c",), 3)


def test_the_same_seed_gives_the_same_boosting(reviewed: Reviewed) -> None:
    a, spec = fitted("boosting", reviewed)
    b, _ = fitted("boosting", reviewed)
    X = models.design(reviewed.frame, spec)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_numbers_are_standardised_before_the_logistic_regression() -> None:
    """The ridge penalty makes a column of tiny numbers worth nothing next to one of huge numbers, unless both are
    put on the same scale first. (The order of the scores of one column does not change with its scale, so it takes
    two columns to see it.)"""
    n = 400
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n)
    y = (signal + rng.normal(size=n) > 0).astype(int)  # the tiny column knows about 80% of the order of y
    frame = pd.DataFrame({"tiny": signal * 1e-4, "huge": rng.normal(size=n) * 1e4})
    spec = models.FeatureSpec("two", ("tiny", "huge"))
    pipeline = models.build_pipeline("logistic", spec)
    pipeline.fit(frame, y)
    assert roc_auc_score(y, pipeline.predict_proba(frame)[:, 1]) > 0.7
