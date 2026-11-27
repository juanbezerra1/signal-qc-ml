"""The models, and the columns each of them may read."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import columns as cols

MODELS = ("prior", "logistic", "boosting")


@dataclass(frozen=True)
class FeatureSpec:
    """A named set of columns: the numbers and the codes that a model is given."""

    name: str
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def __len__(self) -> int:
        return len(self.columns)

    def plus(self, *extra_numeric: str, name: str | None = None) -> FeatureSpec:
        """The same columns and some more numbers (the planted predictor of the controls, say)."""
        return FeatureSpec(name or self.name, self.numeric + tuple(extra_numeric), self.categorical)


def spec_for(steps: Sequence[str], columns: Sequence[str], name: str | None = None) -> FeatureSpec:
    """The columns of the named groups (``cols.STEPS`` and ``review``), as found in a table with ``columns``."""
    numeric: list[str] = []
    categorical: list[str] = []
    for step in steps:
        group = cols.group_named(step, columns)
        numeric.extend(c for c in group.numeric if c not in numeric)
        categorical.extend(c for c in group.categorical if c not in categorical)
    label = name or " + ".join(steps)
    return FeatureSpec(label, tuple(numeric), tuple(categorical))


def design(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """The columns of ``spec`` from ``frame``. Codes (a device, an operator) become text, to be one-hot encoded."""
    if not spec.columns:  # the prior reads nothing, but a model needs a column to hold on to
        return pd.DataFrame({"nothing": np.zeros(len(frame))}, index=frame.index)
    X = frame[list(spec.columns)].copy()
    for column in spec.categorical:
        X[column] = X[column].astype(str)
    return X


def build_pipeline(model: str, spec: FeatureSpec, seed: int = 0) -> BaseEstimator:
    """An unfitted pipeline: the preparation of the columns, then the model.

    * ``prior``: says the share of recordings that needed attention in the training data, for every recording. It
      cannot tell two recordings apart, so its AUC is 0.5 by construction, and a pipeline that scores anything
      else on it is broken.
    * ``logistic``: the columns standardised (a column that does not exist for a modality is set to the mean, that
      is 0), codes one-hot, and a logistic regression with the default ridge penalty.
    * ``boosting``: a small histogram gradient-boosting classifier (depth 3, 100 rounds, a slow learning rate), which
      reads a missing number as missing.
    """
    if model == "prior":
        return Pipeline([("model", DummyClassifier(strategy="prior"))])
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if model == "logistic":
        numbers: BaseEstimator | str = Pipeline(
            [
                ("scale", StandardScaler()),
                ("fill", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
            ]
        )
        estimator: BaseEstimator = LogisticRegression(C=1.0, max_iter=2000)
    elif model == "boosting":
        numbers = "passthrough"
        estimator = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=100,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model {model!r}; the models are {', '.join(MODELS)}")
    prepare = ColumnTransformer(
        [("numbers", numbers, list(spec.numeric)), ("codes", one_hot, list(spec.categorical))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("prepare", prepare), ("model", estimator)])
