"""The evaluation: how well does each set of columns predict the result of the review, on participants never seen?"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, brier_score_loss

from . import columns as cols
from . import cv, stats
from .data import Reviewed
from .models import FeatureSpec, spec_for

#: The question that is answered before looking at any number: one model, all the columns known before the review.
PRIMARY_MODEL = "logistic"


@dataclass(frozen=True)
class Design:
    """How the evaluation is done. Everything that is random takes its seed from here."""

    folds: int = 5
    repeats: int = 10
    seed: int = 0
    bootstrap: int = 1000
    permutations: int = 500

    def as_dict(self) -> dict[str, int]:
        return {
            "folds": self.folds,
            "repeats": self.repeats,
            "seed": self.seed,
            "bootstrap": self.bootstrap,
            "permutations": self.permutations,
        }


DEFAULT_DESIGN = Design()


@dataclass(frozen=True)
class Row:
    """The result of one model on one set of columns."""

    name: str
    model: str
    features: str
    n_features: int
    auc: float
    ci_low: float
    ci_high: float
    average_precision: float
    brier: float
    known_before_review: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "features": self.features,
            "n_features": self.n_features,
            "auc": self.auc,
            "auc_ci95": [self.ci_low, self.ci_high],
            "average_precision": self.average_precision,
            "brier": self.brier,
            "known_before_review": self.known_before_review,
        }

    @property
    def interval_has_chance(self) -> bool:
        """True when 0.5, the AUC of a coin, is inside the 95% interval of the AUC."""
        return not (self.ci_low > 0.5 or self.ci_high < 0.5)


@dataclass(frozen=True)
class Primary:
    """The one test that is stated before looking at any number: is the AUC of ``PRIMARY_MODEL`` on all the columns
    known before the review above what moving the results between participants gives?"""

    auc: float  # of the first pass of cross-validation: the statistic that the moves are compared with
    null_mean: float
    null_sd: float
    critical_95: float  # the AUC that 95% of the moved tables stay under
    p_value: float
    permutations: int
    null: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": PRIMARY_MODEL,
            "auc_first_pass": self.auc,
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "critical_95": self.critical_95,
            "p_value": self.p_value,
            "permutations": self.permutations,
        }


@dataclass
class Evaluation:
    rows: list[Row] = field(default_factory=list)
    primary: Primary | None = None
    predictions: dict[str, cv.Predictions] = field(default_factory=dict)  # the out-of-fold scores, by row name

    def row(self, features: str, model: str | None = None) -> Row:
        """The row of the table for a set of columns (and a model, when more than one has it)."""
        for r in self.rows:
            if r.features == features and (model is None or r.model == model):
                return r
        raise KeyError(f"no row for {features!r}")


def step_specs(columns: list[str]) -> list[FeatureSpec]:
    """The sets of columns that grow one group at a time: acquisition, then the session, the device, ..."""
    out: list[FeatureSpec] = []
    for i, step in enumerate(cols.STEPS):
        name = step if i == 0 else f"+ {step}"
        spec = spec_for(cols.STEPS[: i + 1], columns, name=name)
        if len(spec) and (i == 0 or len(spec) > len(out[-1])):
            out.append(spec)
    return out


BEFORE_REVIEW_NAME = "everything known before the review"
REVIEW_ONLY_NAME = "only what the review produces"
BOTH_NAME = "everything known before + what the review produces"


def before_review(columns: list[str]) -> FeatureSpec:
    """Every column that is known before anyone reviews the recording."""
    return spec_for(cols.STEPS, columns, name=BEFORE_REVIEW_NAME)


def review_only(columns: list[str]) -> FeatureSpec:
    """The columns that the review itself produces: the answer, in other words."""
    return spec_for(["review"], columns, name=REVIEW_ONLY_NAME)


def known_and_review(columns: list[str]) -> FeatureSpec:
    return spec_for([*cols.STEPS, "review"], columns, name=BOTH_NAME)


def _mean_over_folds(
    metric: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, predictions: cv.Predictions
) -> float:
    """The mean of ``metric`` over the folds of all the passes (a fold with one class only does not count)."""
    values = []
    for score, fold in zip(predictions.score, predictions.fold, strict=True):
        for k in np.unique(fold):
            rows = fold == k
            if 0 < y[rows].sum() < rows.sum():
                values.append(metric(y[rows], score[rows]))
    return float(np.mean(values)) if values else float("nan")


def evaluate_one(
    model: str,
    spec: FeatureSpec,
    data: Reviewed,
    splits: list[cv.Split],
    design: Design,
    jobs: int = 1,
    known_before_review: bool = True,
) -> tuple[Row, cv.Predictions]:
    """Score ``model`` on ``spec`` over all the ``splits``; the row of the table and the scores of every pass."""
    predictions = cv.out_of_fold(model, spec, data.frame, data.y, splits, design.seed, jobs)
    folded = stats.FoldedAUC(predictions.score, predictions.fold)
    ci_low, ci_high = stats.participant_bootstrap(data.y, folded, data.groups, design.bootstrap, design.seed)
    row = Row(
        name=f"{model}: {spec.name}",
        model=model,
        features=spec.name,
        n_features=len(spec),
        auc=folded.auc(data.y),
        ci_low=ci_low,
        ci_high=ci_high,
        average_precision=_mean_over_folds(average_precision_score, data.y, predictions),
        brier=float(np.mean([brier_score_loss(data.y, s) for s in predictions.score])),
        known_before_review=known_before_review,
    )
    return row, predictions


def single_pass_auc(
    model: str,
    spec: FeatureSpec,
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> float:
    """The AUC of one pass of grouped cross-validation: the statistic of the permutation test and of the controls."""
    splits = cv.grouped_splits(y, groups, folds, 1, seed)
    predictions = cv.out_of_fold(model, spec, frame, y, splits, seed)
    return stats.FoldedAUC(predictions.score, predictions.fold).auc(y)


def _moved(model: str, spec: FeatureSpec, data: Reviewed, folds: int, seed: int, permutation_seed: int) -> float:
    rng = np.random.default_rng(permutation_seed)
    moved = cv.permute_between(data.y, data.groups, rng)
    return single_pass_auc(model, spec, data.frame, moved, data.groups, folds, seed)


def permutation_test(spec: FeatureSpec, data: Reviewed, design: Design, jobs: int = 1) -> Primary:
    """Compare the AUC of the first pass with the AUCs of the same procedure on results moved between participants.

    Moving the results of a whole participant to another participant who has as many recordings keeps how often each
    participant's recordings need attention and breaks any link between the columns and the result, which is what
    "nothing predicts it" means. The AUC of cross-validation on such results is not exactly 0.5, and the moves show
    where it is: they are the reference, not 0.5.
    """
    observed = single_pass_auc(PRIMARY_MODEL, spec, data.frame, data.y, data.groups, design.folds, design.seed)
    children = np.random.SeedSequence(design.seed).spawn(design.permutations)
    seeds = [int(child.generate_state(1)[0]) for child in children]
    null = np.array(
        Parallel(n_jobs=jobs)(delayed(_moved)(PRIMARY_MODEL, spec, data, design.folds, design.seed, s) for s in seeds)
    )
    return Primary(
        auc=observed,
        null_mean=float(np.nanmean(null)),
        null_sd=float(np.nanstd(null, ddof=1)),
        critical_95=float(np.nanquantile(null, 0.95)),
        p_value=stats.p_value(observed, null),
        permutations=design.permutations,
        null=null,
    )


def evaluate(data: Reviewed, design: Design = DEFAULT_DESIGN, jobs: int = 1, boosting: bool = True) -> Evaluation:
    """The table of results: the prior, the columns added one group at a time, boosting, and the review columns."""
    columns = list(data.frame.columns)
    splits = cv.grouped_splits(data.y, data.groups, design.folds, design.repeats, design.seed)
    result = Evaluation()

    def add(model: str, spec: FeatureSpec, known: bool = True) -> None:
        row, predictions = evaluate_one(model, spec, data, splits, design, jobs, known)
        result.rows.append(row)
        result.predictions[row.name] = predictions

    add("prior", FeatureSpec("nothing"))
    for spec in step_specs(columns):
        add("logistic", spec)
    known = before_review(columns)
    if boosting:
        add("boosting", known)
    add("logistic", review_only(columns), known=False)
    add("logistic", known_and_review(columns), known=False)
    if boosting:
        add("boosting", known_and_review(columns), known=False)
    if design.permutations:
        result.primary = permutation_test(known, data, design, jobs)
    return result
